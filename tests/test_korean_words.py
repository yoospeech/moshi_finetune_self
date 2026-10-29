import json
from pathlib import Path

import pytest

from finetune.korean import KoreanVocabulary
from finetune.korean_words import KoreanWordAlignment, word_alignment_tokens, verify_tokenizer_files


def test_original_word_times_no_model(monkeypatch, tmp_path):
    import whisper
    from whisper.tokenizer import get_tokenizer
    def forbid(*args, **kwargs):
        raise AssertionError('No Whisper speech model may be loaded')
    monkeypatch.setattr(whisper, 'load_model', forbid)
    tokenizer = get_tokenizer(True, language='ko')
    data = dict(alignments=[['안녕하세요', [1., 1.5], 'SPEAKER_MAIN'], ['123!', [2., 2.5], 'SPEAKER_MAIN']])
    tokens = word_alignment_tokens(data, tokenizer)
    assert [t['whisper_token_id'] for t in tokens] == tokenizer.encode('안녕하세요 123!')
    assert all((t['start'], t['end']) in {(1., 1.5), (2., 2.5)} for t in tokens)
    vocabulary = KoreanVocabulary(32000, sorted({t['whisper_token_id'] for t in tokens}))
    import wave
    path = tmp_path / 'header.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(4 * 16000 * 4))
    output = KoreanWordAlignment(vocabulary)(str(path), data)
    assert output['alignment_method'] == 'existing_word_intervals_fifo'
    assert 'tokens' not in data


def test_hf_medium_files_match_native_whisper():
    from whisper.tokenizer import get_tokenizer
    directory = Path('tokenizers/whisper-medium')
    if not directory.exists():
        pytest.skip('Downloaded tokenizer assets not present')
    assert set(verify_tokenizer_files(directory, get_tokenizer(True, language='ko'))) == {'vocab.json', 'merges.txt'}


def test_segment_boundary_and_recording_tail():
    from finetune.korean_words import schedule_word_tokens
    from finetune.data.interleaver import Interleaver
    raw = [dict(whisper_token_id=1, start=199.92, end=200.2, speaker="SPEAKER_MAIN") for _ in range(3)]
    tokens = schedule_word_tokens(raw, 201)
    v = KoreanVocabulary(16, [1])
    inter = Interleaver(None, 12.5, 3, 0, -1, device="cpu", korean_vocabulary=v)
    first = inter.prepare_korean(tokens, 0, 2500)
    second = inter.prepare_korean(tokens, 200, 13)
    assert first.eq(17).sum() == 1
    assert second.eq(17).sum() == 2
    assert inter.korean_stats["dropped"] == 0
    tail = schedule_word_tokens(raw, 200)
    assert [t["moshi_frame"] for t in tail] == [2497, 2498, 2499]
    assert all(t["start"] == 199.92 for t in tail)
    assert inter.prepare_korean(tail, 0, 2500).eq(17).sum() == 3
