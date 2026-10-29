"""Whisper BPE over existing word alignments; no speech-model inference."""
import hashlib
import json
import math
from pathlib import Path


def verify_tokenizer_files(directory, tokenizer):
    """Verify HF vocab + merges exactly match the installed Whisper BPE ranks.

    Execution uses OpenAI's tokenizer, whose rank table encodes the same merges.
    Verification avoids adding a Transformers dependency to the official trainer.
    """
    directory = Path(directory)
    vocab = json.loads((directory / "vocab.json").read_text())
    byte_values = list(range(ord('!'), ord('~') + 1)) + list(range(161, 173)) + list(range(174, 256))
    chars = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            chars.append(256 + extra)
            extra += 1
    inverse = {chr(char): byte for byte, char in zip(byte_values, chars)}
    ordinary = {piece: token for piece, token in vocab.items() if token < tokenizer.eot}
    if len(ordinary) != tokenizer.eot:
        raise ValueError("Incomplete Whisper vocabulary")
    for piece, token in ordinary.items():
        if bytes(inverse[c] for c in piece) != tokenizer.encoding.decode_single_token_bytes(token):
            raise ValueError(f"Whisper vocabulary ID mismatch at {token}")
    merges = (directory / "merges.txt").read_text().splitlines()
    pairs = [line.split() for line in merges if line and not line.startswith('#')]
    for rank, pair in enumerate(pairs, start=256):
        if len(pair) != 2 or ordinary.get(''.join(pair)) != rank:
            raise ValueError(f"Whisper BPE merge mismatch at rank {rank}")
    if len(pairs) != sum(len(piece) > 1 for piece in ordinary):
        raise ValueError("Incomplete Whisper merge rules")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("vocab.json", "merges.txt")}


def word_alignment_tokens(data, tokenizer):
    if "alignments" not in data:
        raise ValueError("existing_words mode requires alignments")
    tokens = []
    previous_speaker = None
    for word, interval, speaker in sorted(data["alignments"], key=lambda item: item[1][0]):
        start, end = map(float, interval)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
            raise ValueError("Invalid existing word timestamp")
        word = word.strip()
        if not word:
            continue
        # Preserve normal whitespace BPE rather than concatenating words on decode.
        text = (" " if speaker == previous_speaker else "") + word
        previous_speaker = speaker
        for token in tokenizer.encode(text):
            tokens.append(dict(whisper_token_id=token, start=start, end=end, speaker=speaker))
    return tokens


def schedule_word_tokens(tokens, duration=None):
    """Global FIFO before slicing segments, retaining original word timestamps."""
    next_frame, requested = {}, {}
    result = []
    for token in tokens:
        speaker = token["speaker"]
        frame = math.floor(token["start"] * 12.5 + 1e-8)
        used = requested.setdefault(speaker, set())
        placed = max(frame, next_frame.get(speaker, 0))
        result.append({**token, "moshi_frame": placed,
                       "frame_collision": frame in used,
                       "frame_shifted": placed != frame})
        used.add(frame)
        next_frame[speaker] = placed + 1
    if duration is not None:
        limit = math.ceil(duration * 12.5)
        # At the recording end only, fit the queue backwards into available
        # frames. Original word timestamps are unchanged; displacement is counted.
        for speaker in next_frame:
            following = limit
            for token in reversed(result):
                if token["speaker"] != speaker:
                    continue
                token["moshi_frame"] = min(token["moshi_frame"], following - 1)
                following = token["moshi_frame"]
                token["frame_shifted"] = following != math.floor(token["start"] * 12.5 + 1e-8)
                if following < 0:
                    raise ValueError("BPE token count exceeds recording frame capacity")
    return result


class KoreanWordAlignment:
    """Attach word intervals to BPE pieces; FIFO interleaving places the pieces."""
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.tokenizer = vocabulary.tokenizer()

    def __call__(self, audio_path, data):
        if "tokens" in data:
            return data
        import wave
        with wave.open(str(audio_path), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        tokens = schedule_word_tokens(word_alignment_tokens(data, self.tokenizer), duration)
        missing = {t["whisper_token_id"] for t in tokens} - self.vocabulary.mapping.keys()
        if missing:
            raise ValueError(f"Existing word alignments contain {len(missing)} unmapped BPE IDs")
        return {**data, "tokens": tokens, "num_languages": self.vocabulary.num_languages,
                "alignment_method": "existing_word_intervals_fifo"}
