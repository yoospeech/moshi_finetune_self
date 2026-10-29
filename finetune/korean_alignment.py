"""Whisper cross-attention + DTW BEFORE word aggregation (OpenAI MIT timing.py)."""
import numpy as np
import torch


def token_boundaries(text_indices, time_indices, count):
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    rows = text_indices[jumps]
    if not np.array_equal(rows, np.arange(count + 1)):
        raise ValueError("DTW did not visit all token rows including EOT boundary")
    return time_indices[jumps] / 50.0


@torch.no_grad()
def align_tokens(model, tokenizer, text, audio, offset_sec=0.0):
    """audio: one 16kHz mono segment <=30s; offset relative to original recording.

    Decoder input is transcript teacher forcing. Whisper embeddings are used
    internally by the aligner only and are NEVER returned or copied into Moshi.
    """
    from whisper.audio import log_mel_spectrogram, pad_or_trim
    from whisper.model import disable_sdpa
    from whisper.timing import dtw, median_filter
    ids = tokenizer.encode(text)
    if not ids:
        return []
    if audio.ndim != 1 or not 0 < len(audio) <= 480000 or offset_sec < 0:
        raise ValueError("Expected a nonempty <=30s mono 16kHz window and positive offset")
    decoder_ids = [*tokenizer.sot_sequence, tokenizer.no_timestamps, *ids, tokenizer.eot]
    if len(decoder_ids) > model.dims.n_text_ctx:
        raise ValueError("Transcript exceeds Whisper context; split aligned windows")
    mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=model.dims.n_mels).to(model.device)
    qks = [None] * model.dims.n_text_layer
    hooks = []
    try:
        for i, block in enumerate(model.decoder.blocks):
            hooks.append(block.cross_attn.register_forward_hook(
                lambda module, ins, outs, index=i: qks.__setitem__(index, outs[-1][0])))
        with disable_sdpa():
            model(mel[None], torch.tensor(decoder_ids, device=model.device)[None])
    finally:
        for hook in hooks:
            hook.remove()
    weights = torch.stack([qks[int(layer)][int(head)] for layer, head in model.alignment_heads.indices().T])
    weights = weights[..., :max(1, len(audio) // 320)].float().softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = median_filter((weights - mean) / std.clamp_min(1e-6), 7)
    matrix = weights.mean(0)[len(tokenizer.sot_sequence):-1]
    if not torch.isfinite(matrix).all():
        raise ValueError("Whisper attention contains nonfinite values")
    ti, times = dtw(-matrix)
    boundaries = token_boundaries(ti, times, len(ids)) + offset_sec
    return [dict(whisper_token_id=token, start=float(boundaries[i]), end=float(boundaries[i + 1])) for i, token in enumerate(ids)]
