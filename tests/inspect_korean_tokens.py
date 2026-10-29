"""Inspect current Korean vocabulary tokens without loading a speech model.

Run from the repository root:
    python tests/inspect_korean_tokens.py
    python tests/inspect_korean_tokens.py '레몬 즙 주세요,'
"""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finetune.korean import KoreanVocabulary


def inspect(text, vocab):
    tokenizer = vocab.tokenizer()
    ids = tokenizer.encode(text)
    print("Whisper IDs:", ids)
    print("모델 IDs:", [vocab.mapping.get(i, "누락") for i in ids])
    for token in ids:
        raw = tokenizer.encoding.decode_single_token_bytes(token)
        print(f"  {token:5d}  bytes={raw!r}")

    missing = sorted(set(ids) - vocab.mapping.keys())
    print("누락 토큰:", missing)
    if missing:
        return False
    restored = vocab.decode(vocab.encode(text), helium=None)
    print("복원:", restored)
    print("원문 일치:", restored == text)
    return restored == text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="생략하면 대화형 입력")
    parser.add_argument(
        "--vocab", type=Path, default=ROOT / "korean_vocab_medium_words.json"
    )
    args = parser.parse_args()
    vocab = KoreanVocabulary.load(args.vocab)
    if args.text is not None:
        return 0 if inspect(args.text, vocab) else 1

    while True:
        try:
            text = input("\n문장 입력 (종료: 엔터): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            return 0
        inspect(text, vocab)


if __name__ == "__main__":
    sys.exit(main())
