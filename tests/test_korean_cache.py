import json
from types import SimpleNamespace

import pytest

from finetune.korean import KoreanVocabulary
from finetune.korean_cache import KoreanAlignmentCache, ensure_vocabulary, selected_turns


def test_lazy_cache_reuse_and_invalidation(tmp_path, monkeypatch):
    audio = tmp_path / 'recording.wav'
    audio.write_bytes(b'fixture')
    cache = KoreanAlignmentCache(tmp_path / 'cache', KoreanVocabulary(16, [1]))
    calls = []
    def align(path, turns):
        calls.append(turns)
        return [dict(whisper_token_id=1, start=turns[0]['start'], end=turns[0]['end'])]
    monkeypatch.setattr(cache, '_align', align)
    data = dict(turns=[dict(speaker='ASSISTANT', text='안녕', start=1., end=2.)], text_conditions={'x': 'y'})
    first = cache(str(audio), data)
    assert cache(str(audio), data) == first
    assert len(calls) == 1
    assert first['text_conditions'] == data['text_conditions']
    assert 'tokens' not in data
    data['turns'][0]['text'] = '안녕하세요'
    cache(str(audio), data)
    assert len(calls) == 2
    assert cache(str(audio), first) is first


def test_auto_vocab_and_mapping_protection(tmp_path):
    sidecar = tmp_path / 'recording.json'
    sidecar.write_text(json.dumps(dict(turns=[dict(speaker='ASSISTANT', text='한국어 123!', start=0., end=2.)])))
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(json.dumps(dict(path=str(sidecar.with_suffix('.wav')), duration=2.))+'\n')
    args = SimpleNamespace(korean_vocab=str(tmp_path/'vocab.json'), whisper_num_languages=99,
                           korean_vocab_include_eval=False, korean_speaker='ASSISTANT',
                           data=SimpleNamespace(train_data=str(manifest), eval_data=''))
    vocabulary = ensure_vocabulary(args, 16)
    assert vocabulary.encode('한국어 123!')
    original = (tmp_path/'vocab.json').read_bytes()
    ensure_vocabulary(args, 16)
    assert (tmp_path/'vocab.json').read_bytes() == original
    with pytest.raises(ValueError, match='incompatible'):
        ensure_vocabulary(args, 17)


def test_turn_validation():
    with pytest.raises(ValueError, match='requires turns'):
        selected_turns({'alignments': []})
    with pytest.raises(ValueError, match='30 seconds'):
        selected_turns(dict(turns=[dict(speaker='ASSISTANT', start=0, end=31, text='x')]))
