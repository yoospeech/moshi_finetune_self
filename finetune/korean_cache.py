"""Lazy, persistent token alignment from existing turn annotations."""
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path

from .korean import KoreanVocabulary

logger = logging.getLogger(__name__)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def selected_turns(data, speaker="ASSISTANT"):
    if "turns" not in data:
        raise ValueError("Automatic alignment requires turns with speaker/text/start/end")
    result = []
    for turn in data["turns"]:
        if turn["speaker"] != speaker:
            continue
        start, end = float(turn["start"]), float(turn["end"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end or end - start > 30:
            raise ValueError(f"Whisper turn must be finite, positive and <=30 seconds: {start}, {end}")
        if turn["text"].strip():
            result.append(turn)
    return sorted(result, key=lambda t: t["start"])


def ensure_vocabulary(args, base_card):
    """Run on rank zero before allocating Moshi. Never rewrite an existing mapping."""
    from .data.dataset import parse_data_sources
    from whisper.tokenizer import get_tokenizer
    if not args.korean_vocab:
        raise ValueError("korean_vocab is required with korean_auto_prepare")
    tokenizer = get_tokenizer(True, num_languages=args.whisper_num_languages, language="ko", task="transcribe")
    mode = getattr(args, "korean_alignment_mode", "whisper_dtw")
    if mode not in {"existing_words", "whisper_dtw"}:
        raise ValueError("korean_alignment_mode must be existing_words or whisper_dtw")
    tokenizer_files = getattr(args, "whisper_tokenizer_dir", None)
    fingerprints = None
    if tokenizer_files:
        from .korean_words import verify_tokenizer_files
        fingerprints = verify_tokenizer_files(tokenizer_files, tokenizer)
    ids = set()
    sources = [args.data.train_data]
    if args.korean_vocab_include_eval and args.data.eval_data:
        sources.append(args.data.eval_data)
    record_count = 0
    for source in sources:
        datasets, _ = parse_data_sources(source)
        for dataset in datasets:
            for manifest in sorted(dataset.jsonl_files):
                for line in manifest.read_text().splitlines():
                    record = json.loads(line)
                    data = json.loads(Path(record["path"]).with_suffix(".json").read_text())
                    if "tokens" in data:
                        if data.get("num_languages", args.whisper_num_languages) != args.whisper_num_languages:
                            raise ValueError("Whisper tokenizer mismatch in sidecar")
                        ids.update(t["whisper_token_id"] for t in data["tokens"])
                    elif mode == "existing_words":
                        from .korean_words import word_alignment_tokens, schedule_word_tokens
                        tokens = schedule_word_tokens(word_alignment_tokens(data, tokenizer), record["duration"])
                        ids.update(t["whisper_token_id"] for t in tokens)
                    else:
                        for turn in selected_turns(data, args.korean_speaker):
                            encoded = tokenizer.encode(turn["text"])
                            if len(encoded) + len(tokenizer.sot_sequence) + 2 > 448:
                                raise ValueError("Turn exceeds Whisper text context; split the annotation")
                            ids.update(encoded)
                    record_count += 1
    path = Path(args.korean_vocab)
    if path.exists():
        vocabulary = KoreanVocabulary.load(path)
        if vocabulary.base_card != base_card or vocabulary.num_languages != args.whisper_num_languages:
            raise ValueError("Existing vocabulary is incompatible with model/tokenizer")
        missing = ids - set(vocabulary.whisper_ids)
        if missing:
            raise ValueError(f"Existing vocabulary lacks {len(missing)} tokens; use a new korean_vocab path")
    else:
        vocabulary = KoreanVocabulary(base_card, sorted(ids), args.whisper_num_languages)
        from dataclasses import asdict
        atomic_json(path, asdict(vocabulary))
    logger.info("Korean vocabulary: %d tokens, %d records, transcript coverage 100%%", len(vocabulary.whisper_ids), record_count)
    atomic_json(path.with_suffix(".preparation.json"), dict(records=record_count, vocabulary_size=len(vocabulary.whisper_ids), transcript_token_coverage=1.0, includes_eval=args.korean_vocab_include_eval, alignment_status=mode, tokenizer_files=fingerprints))
    return vocabulary


class KoreanAlignmentCache:
    def __init__(self, directory, vocabulary, model="small", device="cpu", speaker="ASSISTANT", channel=0):
        self.directory = Path(directory)
        self.vocabulary = vocabulary
        self.model_name = model
        self.device = device
        self.speaker = speaker
        self.channel = channel
        self._model = None
        if channel not in (0, 1):
            raise ValueError("Expected stereo channel 0 or 1")

    def _align(self, path, turns):
        import sphn
        import torch
        import torchaudio
        import whisper
        from .korean_alignment import align_tokens
        if self._model is None:
            logger.info("Loading Whisper %s on %s for lazy alignment", self.model_name, self.device)
            self._model = whisper.load_model(self.model_name, device=self.device).eval()
            self._model.requires_grad_(False)
            if not self._model.is_multilingual or self._model.num_languages != self.vocabulary.num_languages:
                raise ValueError("Whisper model/tokenizer mismatch")
        audio, sr = sphn.read(str(path))
        if audio.shape[0] != 2:
            raise ValueError("Expected original stereo recording")
        audio = torchaudio.functional.resample(torch.from_numpy(audio[self.channel]), sr, 16000)
        tokenizer = self.vocabulary.tokenizer()
        tokens = []
        for turn in turns:
            start, end = round(turn["start"] * 16000), round(turn["end"] * 16000)
            if not 0 <= start < end <= len(audio):
                raise ValueError(f"Turn outside recording: {path}")
            aligned = align_tokens(self._model, tokenizer, turn["text"], audio[start:end], start / 16000)
            for token in aligned:
                token["speaker"] = "SPEAKER_MAIN"
                if token["whisper_token_id"] not in self.vocabulary.mapping:
                    raise ValueError("Aligned token absent from fixed training vocabulary")
            tokens.extend(aligned)
        return sorted(tokens, key=lambda t: t["start"])

    def __call__(self, audio_path, data):
        if "tokens" in data:
            return data
        turns = selected_turns(data, self.speaker)
        path = Path(audio_path).resolve()
        stat = path.stat()
        identity = dict(version=1, audio=str(path), size=stat.st_size, mtime=stat.st_mtime_ns,
                        turns=turns, model=self.model_name, num_languages=self.vocabulary.num_languages,
                        channel=self.channel)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        cache = self.directory / f"{path.stem}-{digest}.json"
        # Multiple training ranks may encounter different chunks of one recording.
        with cache.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache.exists():
                aligned = json.loads(cache.read_text())
            else:
                logger.info("Whisper token alignment: %s (%d turns)", path.name, len(turns))
                aligned = dict(tokens=self._align(path, turns), num_languages=self.vocabulary.num_languages,
                               alignment_method="whisper_cross_attention_dtw_tokens", source=identity)
                atomic_json(cache, aligned)
        return {**data, **aligned}
