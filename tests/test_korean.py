import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from moshi.models.lm import LMModel

from finetune.korean import KoreanVocabulary, expand_vocabulary, load_for_inference, vocabulary_parameter
from finetune.data.interleaver import Interleaver, InterleavedTokenizer
from finetune.korean_alignment import token_boundaries


def tiny():
    return LMModel(delays=[0, 0, 0], n_q=2, dep_q=2, card=8, text_card=16,
                   dim=16, num_heads=2, num_layers=1, hidden_scale=2, context=16,
                   depformer_dim=16, depformer_num_heads=2, depformer_num_layers=1,
                   gating="silu", depformer_weights_per_step=True, existing_text_padding_id=3)


def vocab():
    return KoreanVocabulary(16, [1, 2, 3])


def inter(**kwargs):
    sp = SimpleNamespace(encode=lambda text, **kw: [4] if isinstance(text, str) else [[4] for _ in text], bos_id=lambda: 1, eos_id=lambda: 2)
    return Interleaver(sp, 12.5, 3, 0, -1, device="cpu", korean_vocabulary=vocab(), **kwargs)


def test_preservation_and_gradient():
    model = tiny()
    originals = {k: v.clone() for k, v in model.state_dict().items()}
    expand_vocabulary(model, vocab())
    assert model.text_initial_token_id == 16
    for key, original in originals.items():
        actual = model.state_dict()[key]
        assert torch.equal(actual[:len(original)], original) if actual.ndim else torch.equal(actual, original)
    codes = torch.randint(0, 8, (1, 3, 8))
    codes[:, 0] = 17
    output = model(codes)
    loss = torch.nn.functional.cross_entropy(output.text_logits.flatten(0, 2), codes[:, :1].flatten()) + output.logits.square().mean()
    loss.backward()
    for layer in (model.text_emb, model.depformer_text_emb, model.text_linear):
        assert layer.weight.grad[17].abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    old = model.text_emb.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old[17], model.text_emb.weight[17])


def test_mapping_roundtrip(tmp_path):
    from whisper.tokenizer import get_tokenizer
    text = "안녕하세요 123, OpenAI!"
    ids = get_tokenizer(True, language="ko").encode(text)
    vocabulary = KoreanVocabulary(16, sorted(set(ids)))
    vocabulary.save(tmp_path / "v.json")
    loaded = KoreanVocabulary.load(tmp_path / "v.json")
    assert loaded.mapping == vocabulary.mapping
    assert min(loaded.mapping.values()) == 17
    assert loaded.decode(loaded.encode(text), None) == text
    with pytest.raises(KeyError):
        vocab().encode(text)


def test_alignment_and_overflow():
    i = inter()
    ts = [dict(whisper_token_id=1, start=10.4, end=10.56), dict(whisper_token_id=2, start=10.4, end=10.5)]
    stream = i.prepare_korean(ts, 10, 10)
    assert stream.shape == (1, 1, 10)
    assert stream[0, 0, 5:7].tolist() == [17, 18]
    assert i.korean_stats == dict(tokens=2, collisions=1, shifted=1, dropped=0, filtered=0)
    with pytest.raises(ValueError, match="capacity"):
        i.prepare_korean(ts, 10, 6)
    assert i.korean_stats["dropped"] == 1
    assert i.prepare_korean(ts, 11, 10).eq(3).all()
    assert np.allclose(token_boundaries(np.array([0, 0, 1, 2]), np.array([0, 1, 2, 3]), 2), [0, .04, .06])


def test_english_regression():
    i = inter()
    assert i.prepare_item([("hello", (.16, .32), "SPEAKER_MAIN")], .8).flatten().tolist() == [3, 0, 4, 3, 3, 3, 3, 3, 3, 3]


def test_stereo_shapes(tmp_path):
    class Mimi:
        frame_rate = 12.5
        def encode(self, x):
            return torch.zeros(2, 2, 5, dtype=torch.long)
    path = tmp_path / "audio.wav"
    path.with_suffix(".json").write_text(json.dumps({"tokens": [dict(whisper_token_id=1, start=.08, end=.16)]}))
    sample = InterleavedTokenizer(Mimi(), inter(), .8)(np.zeros((2, 100)), 0, str(path))
    assert sample.codes.shape == (1, 5, 10)
    assert sample.codes[..., 5:].eq(-1).all()


@pytest.mark.parametrize("adapter", [False, True])
def test_checkpoint_inference(tmp_path, adapter, monkeypatch):
    from moshi.modules.lora import replace_all_linear_with_lora
    from finetune.korean import install_vocabulary
    from finetune.checkpointing import Checkpointer
    import finetune.checkpointing as cp
    monkeypatch.setattr(cp, "get_rank", lambda: 0)
    monkeypatch.setattr(cp, "get_world_size", lambda: 1)
    model = tiny()
    base = {k: v.clone() for k, v in model.state_dict().items()}
    expand_vocabulary(model, vocab())
    if adapter:
        replace_all_linear_with_lora(model, 2, 1.)
        install_vocabulary(model, vocab())
        for name, param in model.named_parameters():
            param.requires_grad = "lora" in name or vocabulary_parameter(name)
    config = dict(lora_rank=2, lora_scaling=1.)
    checkpointer = Checkpointer(model, None, tmp_path, config, korean_vocabulary=vocab())
    states = checkpointer.retrieve_save_states(adapter, torch.float32)
    import safetensors.torch
    safetensors.torch.save_file(states, tmp_path / ("lora.safetensors" if adapter else "consolidated.safetensors"))
    checkpointer.write_params_info(tmp_path)
    def get(**kw):
        model = tiny()
        model.load_state_dict(base)
        return model
    loaded, vocabulary = load_for_inference(SimpleNamespace(get_moshi=get), tmp_path)
    assert loaded.text_initial_token_id == 16
    for key, value in states.items():
        assert torch.equal(loaded.state_dict()[key], value)
    assert loaded.text_linear(torch.zeros(1, 16))[..., 16].isneginf().all()


def test_real_whisper_attention_alignment():
    from whisper.model import Whisper, ModelDimensions
    from whisper.tokenizer import get_tokenizer
    from finetune.korean_alignment import align_tokens
    model = Whisper(ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=16,
                    n_audio_head=2, n_audio_layer=2, n_vocab=51865,
                    n_text_ctx=64, n_text_state=16, n_text_head=2, n_text_layer=2)).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=.02)
    tokenizer = get_tokenizer(True, language="ko")
    text = "한국어!"
    tokens = align_tokens(model, tokenizer, text, torch.randn(16000), offset_sec=10)
    assert [t["whisper_token_id"] for t in tokens] == tokenizer.encode(text)
    assert all(10 <= t["start"] <= t["end"] <= 11 for t in tokens)
    assert all(not b.cross_attn._forward_hooks for b in model.decoder.blocks)


def test_lora_gradient_and_consolidation(tmp_path, monkeypatch):
    from moshi.modules.lora import replace_all_linear_with_lora
    from finetune.korean import install_vocabulary
    import finetune.checkpointing as cp
    monkeypatch.setattr(cp, "get_rank", lambda: 0)
    monkeypatch.setattr(cp, "get_world_size", lambda: 1)
    model = tiny()
    expand_vocabulary(model, vocab())
    replace_all_linear_with_lora(model, 2, 1.)
    install_vocabulary(model, vocab())
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name or vocabulary_parameter(name)
    codes = torch.ones(1, 3, 5, dtype=torch.long)
    codes[:, 0] = 17
    output = model(codes)
    loss = torch.nn.functional.cross_entropy(output.text_logits.flatten(0, 2), codes[:, :1].flatten()) + output.logits.square().mean()
    loss.backward()
    for parameter in (model.text_emb.weight, model.depformer_text_emb.weight, model.text_linear.frozen_W.weight):
        assert parameter.grad[17].abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    states = cp.Checkpointer(model, None, tmp_path, {}).retrieve_save_states(False, torch.float32)
    restored = tiny()
    expand_vocabulary(restored, vocab())
    restored.load_state_dict(states, strict=True)
    torch.testing.assert_close(restored(codes).text_logits, model(codes).text_logits, atol=1e-5, rtol=1e-5)


def test_lmgen_expanded_tokens():
    from moshi.models.lm import LMGen
    model = tiny().eval()
    expand_vocabulary(model, vocab())
    # Force an expanded token to verify the actual autoregressive Depth path.
    def force(module, inputs, output):
        forced = torch.full_like(output, -torch.inf)
        forced[..., 17] = 0
        return forced
    model.text_linear.register_forward_hook(force)
    generator = LMGen(model, use_sampling=False)
    with generator.streaming(1):
        result = generator.step(torch.zeros(1, 0, 1, dtype=torch.long))
    assert result[0, 0, 0] == 17


def test_meta_expansion():
    with torch.device("meta"):
        model = tiny()
    expand_vocabulary(model, vocab())
    assert model.text_emb.weight.shape == (21, 16)
    assert model.text_initial_token_id == 16


def test_training_loader_lora(tmp_path, monkeypatch):
    import safetensors.torch
    import finetune.wrapped_model as wrapped
    from finetune.args import TrainArgs, LoraArgs
    from finetune.data.args import DataArgs
    base = tiny()
    safetensors.torch.save_file(base.state_dict(), tmp_path / "base.safetensors")
    vocab().save(tmp_path / "vocab.json")
    def get(**kwargs):
        return tiny()
    info = SimpleNamespace(get_moshi=get, moshi_weights=tmp_path / "base.safetensors")
    args = TrainArgs(data=DataArgs(), run_dir=str(tmp_path), param_dtype="float32",
                     korean_vocab=str(tmp_path / "vocab.json"), lora=LoraArgs(enable=True, rank=2))
    monkeypatch.setattr(wrapped, "get_rank", lambda: 0)
    monkeypatch.setattr(wrapped, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    model = wrapped.get_fsdp_model(args, info)
    assert torch.equal(model.text_emb.weight[:17], base.text_emb.weight)
    assert model.text_linear.frozen_W.weight.requires_grad
    assert not model.transformer.layers[0].self_attn.in_projs[0].frozen_W.weight.requires_grad
    assert model.depformer_text_emb.weight.requires_grad
