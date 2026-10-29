"""Prepare sidecars or build/audit vocabulary: python korean_prepare.py --help."""
import argparse
import json
import math
import warnings
from pathlib import Path

from finetune.korean import KoreanVocabulary


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    align = sub.add_parser("align")
    align.add_argument("audio")
    align.add_argument("windows", help="JSON list: start,end,text,channel,speaker; times in original recording")
    align.add_argument("output")
    align.add_argument("--model", default="small")
    align.add_argument("--device", default="cpu")
    vocab = sub.add_parser("vocab")
    vocab.add_argument("manifest", help="Official JSONL path/duration manifest")
    vocab.add_argument("output")
    vocab.add_argument("--base-card", type=int, default=32000)
    vocab.add_argument("--num-languages", type=int, default=99)
    vocab.add_argument("--duration-sec", type=float, default=100)
    vocab.add_argument("--existing", help="Audit coverage against a saved vocabulary")
    args = parser.parse_args()
    if args.command == "align":
        import torch
        import torchaudio
        import whisper
        from whisper.tokenizer import get_tokenizer
        from finetune.korean_alignment import align_tokens
        audio, sr = torchaudio.load(args.audio)
        if audio.shape[0] != 2:
            raise ValueError("Stereo recording required; channel 0 must retain official main-speaker convention")
        audio = torchaudio.functional.resample(audio, sr, 16000)
        model = whisper.load_model(args.model, device=args.device).eval()
        if not model.is_multilingual:
            raise ValueError("Multilingual Whisper required")
        tokenizer = get_tokenizer(True, num_languages=model.num_languages, language="ko", task="transcribe")
        tokens = []
        for window in json.loads(Path(args.windows).read_text()):
            start, end = window["start"], window["end"]
            if not 0 <= start < end <= audio.shape[-1] / 16000 or end - start > 30:
                raise ValueError("Windows must be within audio and at most 30 seconds")
            chunk = audio[window["channel"], round(start * 16000):round(end * 16000)]
            aligned = align_tokens(model, tokenizer, window["text"], chunk, start)
            for token in aligned:
                token["speaker"] = window["speaker"]
            tokens.extend(aligned)
        Path(args.output).write_text(json.dumps({"tokens": sorted(tokens, key=lambda t: t["start"]), "num_languages": model.num_languages}, ensure_ascii=False, indent=2))
        return
    from finetune.data.interleaver import Interleaver
    if args.duration_sec <= 0 or not math.isclose(args.duration_sec * 12.5, round(args.duration_sec * 12.5)):
        raise ValueError("duration-sec must be a positive multiple of 80ms")
    records, ids = [], set()
    for line in Path(args.manifest).read_text().splitlines():
        record = json.loads(line)
        sidecar = json.loads(Path(record["path"]).with_suffix(".json").read_text())
        if sidecar.get("num_languages", args.num_languages) != args.num_languages:
            raise ValueError("Whisper tokenizer language-count mismatch")
        tokens = sidecar["tokens"]
        if any(t["start"] >= record["duration"] or t["end"] > record["duration"] for t in tokens):
            raise ValueError("Token outside original recording")
        ids.update(t["whisper_token_id"] for t in tokens)
        records.append((record, tokens))
    vocabulary = KoreanVocabulary.load(args.existing) if args.existing else KoreanVocabulary(args.base_card, sorted(ids), args.num_languages)
    vocabulary.save(args.output)
    # Statistics count the main speaker, matching training, with all-speaker coverage.
    interleaver = Interleaver(None, 12.5, 3, 0, -1, device="cpu", keep_main_only=True, korean_vocabulary=vocabulary, korean_overflow="warn")
    totals = dict(tokens=0, collisions=0, shifted=0, dropped=0, filtered=0)
    covered = total = 0
    for record, tokens in records:
        total += len(tokens)
        covered += sum(t["whisper_token_id"] in vocabulary.mapping for t in tokens)
        known = [t for t in tokens if t["whisper_token_id"] in vocabulary.mapping]
        for segment in range(math.ceil(record["duration"] / args.duration_sec)):
            start = segment * args.duration_sec
            frames = math.ceil(min(args.duration_sec, record["duration"] - start) * 12.5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                interleaver.prepare_korean(known, start, frames)
            for key in totals:
                totals[key] += interleaver.korean_stats[key]
    duration = sum(r["duration"] for r, _ in records)
    report = dict(korean_vocabulary_size=len(vocabulary.whisper_ids), whisper_token_coverage_rate=covered / max(total, 1), average_korean_tokens_per_second=totals["tokens"] / max(duration, 1e-9), **totals)
    for key, label in (("collisions", "frame_collision_rate"), ("shifted", "shifted_token_rate"), ("dropped", "dropped_token_rate")):
        report[label] = totals[key] / max(totals["tokens"], 1)
    Path(args.output + ".stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if covered != total or totals["dropped"]:
        raise SystemExit("Audit failed: uncovered or dropped tokens; see statistics")


if __name__ == "__main__":
    main()
