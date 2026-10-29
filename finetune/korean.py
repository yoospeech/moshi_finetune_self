"""Separate Whisper BPE vocabulary; no Whisper model weights enter Moshi."""
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import torch
from moshi.models.lm import LMModel


@dataclass
class KoreanVocabulary:
    base_card: int
    whisper_ids: list[int]
    num_languages: int = 99

    def __post_init__(self):
        if self.base_card < 3 or self.whisper_ids != sorted(set(self.whisper_ids)):
            raise ValueError("Expected positive base cardinality and sorted unique IDs")
        tokenizer = self.tokenizer()
        if any(i < 0 or i >= tokenizer.eot for i in self.whisper_ids):
            raise ValueError("Only ordinary multilingual Whisper BPE tokens are allowed")

    def tokenizer(self):
        from whisper.tokenizer import get_tokenizer
        return get_tokenizer(True, num_languages=self.num_languages, language="ko", task="transcribe")

    @cached_property
    def mapping(self):
        # Reserve the original Moshi initial token, including its embedding row.
        return {w: self.base_card + 1 + i for i, w in enumerate(self.whisper_ids)}

    @property
    def text_card(self):
        return self.base_card + 1 + len(self.whisper_ids)

    def encode(self, text):
        mapping = self.mapping
        return [mapping[t] for t in self.tokenizer().encode(text)]

    def decode(self, ids, helium, special_ids=()):
        inverse = {v: k for k, v in self.mapping.items()}
        result, pending, mode = [], [], None
        def flush():
            if pending:
                result.append(self.tokenizer().decode(pending) if mode == "ko" else helium.decode(pending))
                pending.clear()
        for token in ids:
            if token in special_ids or token == self.base_card or token == self.text_card or token < 0:
                flush()
                continue
            current = "ko" if token in inverse else "en"
            if current != mode:
                flush()
                mode = current
            pending.append(inverse[token] if current == "ko" else token)
        flush()
        return "".join(result)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class KoreanLMModel(LMModel):
    @property
    def text_initial_token_id(self):
        return self.korean_base_card


def install_vocabulary(model, vocabulary):
    """Install sentinel semantics on an already expanded official LMModel."""
    if model.text_card != vocabulary.text_card:
        raise ValueError("Model/vocabulary cardinality mismatch")
    if type(model) not in (LMModel, KoreanLMModel):
        raise TypeError("Only the official unquantized LMModel is supported")
    model.__class__ = KoreanLMModel
    model.korean_base_card = vocabulary.base_card
    # Original initial token is an input-only sentinel, never a prediction target.
    def mask_reserved(module, inputs, output):
        output = output.clone()
        output[..., vocabulary.base_card] = -torch.inf
        return output
    if hasattr(model, "_korean_mask_handle"):
        model._korean_mask_handle.remove()
    model._korean_mask_handle = model.text_linear.register_forward_hook(mask_reserved)
    model.korean_vocabulary = vocabulary


def expand_vocabulary(model, vocabulary, seed=0):
    if model.text_card != vocabulary.base_card:
        raise ValueError("Expansion requires the original base vocabulary")
    if model.text_linear.out_features != vocabulary.base_card:
        raise ValueError("Asymmetric text output vocabulary is unsupported")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for name in ("text_emb", "depformer_text_emb", "text_linear"):
        layer = getattr(model, name)
        if layer is None:
            continue
        if not hasattr(layer, "weight"):
            raise ValueError(f"Expand before LoRA wrapping: unsupported layer {name}")
        old = layer.weight
        expected = vocabulary.base_card + (name != "text_linear")
        if old.shape[0] != expected:
            raise ValueError(f"Unsupported embedding layout: {name}")
        count = vocabulary.text_card + (name != "text_linear")
        new = torch.empty((count, *old.shape[1:]), device=old.device, dtype=old.dtype)
        if not old.is_meta:
            # Scalar normal fit per layer, computed in FP32; original rows copied exactly.
            source = old.detach().float().cpu()
            new.copy_((torch.randn(new.shape, generator=generator) * source.std(unbiased=False) + source.mean()).to(old))
            new[:len(old)].copy_(old.detach())
        layer.weight = torch.nn.Parameter(new)
        if name == "text_linear":
            layer.out_features = count
            if layer.bias is not None:
                bias = torch.zeros(count, device=old.device, dtype=old.dtype)
                if not old.is_meta:
                    bias[:expected].copy_(layer.bias.detach())
                layer.bias = torch.nn.Parameter(bias)
        else:
            layer.num_embeddings = count
    model.text_card = vocabulary.text_card
    install_vocabulary(model, vocabulary)


def vocabulary_parameter(name):
    return name.split(".")[0] in {"text_emb", "depformer_text_emb", "text_linear"}


def load_for_inference(checkpoint_info, directory, device="cpu", dtype=torch.float32):
    """Load consolidated weights, or base + adapters including expanded vocabulary."""
    import safetensors.torch
    from moshi.modules.lora import replace_all_linear_with_lora
    directory = Path(directory)
    vocabulary = KoreanVocabulary.load(directory / "korean_vocabulary.json")
    config = json.loads((directory / "config.json").read_text())
    adapter = directory / "lora.safetensors"
    model = checkpoint_info.get_moshi(device=device, dtype=dtype, lm_kwargs_overrides={"lora": False})
    expand_vocabulary(model, vocabulary)
    if adapter.exists():
        replace_all_linear_with_lora(model, config["lora_rank"], config["lora_scaling"])
        # Replacing the output module removes its sentinel hook.
        install_vocabulary(model, vocabulary)
        state = safetensors.torch.load_file(adapter, device=str(device))
        required = {k for k, _ in model.named_parameters() if "lora" in k or vocabulary_parameter(k)}
        if not required.issubset(state):
            raise ValueError("Adapter is missing vocabulary or LoRA parameters")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
    else:
        model.load_state_dict(safetensors.torch.load_file(directory / "consolidated.safetensors", device=str(device)), strict=True)
    return model.eval(), vocabulary
