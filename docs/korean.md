# Korean vocabulary extension

## Current training configuration: reuse existing word times

Set `data.train_data` and `data.eval_data` in
`example/moshi_whisper_korean.yaml` to your local manifests. Before using the
example, download the tokenizer verification assets (not tracked in Git):

```bash
hf download openai/whisper-medium vocab.json merges.txt --local-dir tokenizers/whisper-medium
```

The vocabulary mapping is generated from your manifests by
`korean_auto_prepare: true`. Training outputs and generated mappings are local
artifacts; each checkpoint saves its own mapping for inference.

The active `example/moshi_whisper_korean.yaml` uses `existing_words` mode,
Whisper-medium BPE, and **200-second** segments. This supersedes the DTW/lazy
alignment setup described below for this example. No Whisper speech model is
loaded. Original `alignments` are tokenized with OpenAI's multilingual tokenizer;
the downloaded medium `vocab.json` and `merges.txt` are verified against every
ordinary token ID and merge rank. Both retain the original rules; the expanded
Moshi inventory is the corpus-selected subset, including required shared tokens.

BPE pieces retain their original word start/end timestamps. Text placement is
scheduled FIFO on the full recording's 80ms grid, then sliced into fixed segments.
A token crossing a segment boundary appears once in the next segment. At the
actual recording end, a reverse pass fits the tail into available frames while
preserving order. If the recording has fewer frames than tokens, preparation
raises an error. Placement metadata (`moshi_frame`, `frame_shifted`) is separate
from source timestamps; this is **word alignment reuse**, not token-level DTW.
Padding positions are not dropped tokens. Original WAV/JSON files are untouched.

The active mapping is `korean_vocab_medium_words.json`, separate from the earlier
turn-transcript mapping. Checkpoints must use the mapping saved beside them.
Vocabulary inventory includes train/eval annotations as configured, but only
training examples update weights. The 200-second audit of all 1,000 recordings
found 2,480 vocabulary entries, 100% coverage, and zero dropped or duplicated
tokens. Train: 320,519 tokens, 2.654 tokens/sec, 60.859% requested-frame collisions,
61.176% shifted tokens. Eval: 17,815 tokens, 2.621 tokens/sec, 61.022% collisions,
61.297% shifted tokens. High collision counts are expected because pieces from
one word share its original start. Only one training token was placed earlier
than its requested frame at a recording tail. Exact counts are saved in
`korean_vocab_medium_words.stats.json`. 22 tests pass; a full 200-second GPU
training run has not been executed.


This opt-in path retains the official Temporal/Depth transformers and frozen Mimi.
Whisper provides multilingual BPE IDs and cross-attention/DTW timing only. No
Whisper embedding or prediction weights are transferred to Moshi.

## Prepare data

Keep stereo channel order and speaker labels from the official annotation pipeline.
Supply transcript windows in original recording time, at most 30 seconds each:

```json
[{"start": 0.0, "end": 4.0, "text": "안녕하세요 123!", "channel": 0, "speaker": "SPEAKER_MAIN"}]
```

Use channel 1 and the other speaker label for the other participant. Overlapping
speaker windows are allowed; do not duplicate transcript windows for one speaker.
Window boundaries and transcripts are required inputs: this command does not
transcribe or automatically split recordings. Long recordings remain intact;
Whisper windows are independent of fixed training segments.

```bash
python korean_prepare.py align recording.wav windows.json recording.json --model small --device cuda
python korean_prepare.py vocab train.jsonl korean_vocab.json --base-card 32000 --duration-sec 100
# Audit a validation corpus against the exact training mapping:
python korean_prepare.py vocab validation.jsonl validation_vocab.json --existing korean_vocab.json --duration-sec 100
```

The manifest uses the official `path`/`duration` JSONL schema. Sidecars contain
`tokens: [{whisper_token_id, start, end, speaker}]` and `num_languages`.
Timestamps include the window offset. A missing speaker means SPEAKER_MAIN.
Whisper large-v3 uses a different language count: pass `--num-languages 100`
when building its vocabulary. Ordinary token IDs are validated; special Whisper
tokens are not imported. The vocabulary includes every observed token, including
spaces, digits, punctuation, and Latin characters. Unknown IDs fail at training
and unknown prompt tokens fail at inference; expand/retrain instead of substituting.

Alignment follows OpenAI Whisper's `timing.find_alignment` attention normalization,
median filter and DTW, but returns every token boundary before word grouping.
No word-level timing interpolation occurs. Repeated boundaries (zero-duration BPE
pieces) are valid. Timing is an estimate, not a phonetic guarantee. Silence and
incorrect transcripts can produce inaccurate alignments; review real data.

## Train

Add these fields to an official training YAML; leave model paths pointing to the
original pretrained checkpoint:

```yaml
korean_vocab: /absolute/path/korean_vocab.json
korean_overflow: error
duration_sec: 100
```

Use the normal `torchrun ... train.py --config ...` command. Both full fine-tuning
and LoRA are supported. Mimi is frozen in the existing training entry point.
English sidecars with `alignments` continue through the existing SentencePiece
path. A frame-count-as-seconds bug in the common loader is corrected.

For original vocabulary cardinality V, IDs 0 through V-1 remain unchanged;
ID V remains the input-only text initial sentinel. Sorted Whisper IDs map to
V+1 through V+N. The Temporal embedding, Depth text embedding, and prediction
head expand. Original rows are copied exactly; new rows use a seeded normal
fit to each original layer's FP32 mean and population standard deviation.
The original sentinel is masked from output logits. The unused final input row
is retained for the official model's `text_card+1` layout.

Expansion occurs after loading base weights and before LoRA/FSDP/optimizer setup.
All three vocabulary modules remain trainable under LoRA, including the head's
`frozen_W`. Original rows are preserved **at initialization**, but may train
subsequently. Transformer architecture and pretrained weights are unchanged at
initialization. No loss/objective change is made except excluding the reserved
input-only ID from prediction.

Text uses 12.5Hz frames, floor(start*12.5), stable ordering for equal starts,
and a FIFO forward shift on collisions. Segment ownership is half-open by
start time. Tokens extending across a boundary occur once in their start segment.
No queue crosses training segments; overflow raises an error by default.
`korean_overflow: warn` explicitly permits drops with a warning and counters.
Long silence retains padding and the existing end-of-padding marker. Main-speaker
filtering is counted separately from capacity drops. Overlapping main-speaker
tokens use the same FIFO policy; other-speaker tokens are intentionally filtered.
Segments must be multiples of 80ms. Delay and turn BOS/EOS options are rejected
on the Korean path rather than silently changing alignment semantics.

## Statistics

`vocab` writes `<output>.stats.json`: vocabulary size, occurrence-weighted coverage,
main-speaker tokens per original recording second, frame-collision rate,
shifted-token rate, dropped-token rate, and raw counts. Rate denominators are
selected covered main-speaker tokens; collisions count repeated requested frames,
shifts count all displaced tokens (including eventual drops). Unknown tokens
reduce coverage and are excluded from placement statistics. Nonzero drops or
incomplete coverage cause a failing exit status after writing the report.
Use the same segment duration as training. Training exposes per-segment counts
on `interleaver.korean_stats`. Real corpus statistics require an actual manifest;
no Korean corpus was supplied with this implementation.

## Checkpoints and inference

Both adapter and merged checkpoints include `korean_vocabulary.json` beside
`config.json`. Adapters contain all vocabulary weights, not just LoRA matrices.
Use the helper with CheckpointInfo describing the **original base checkpoint**:

```python
from finetune.korean import load_for_inference
from moshi.models.lm import LMGen
model, vocabulary = load_for_inference(base_checkpoint_info, checkpoint_directory,
                                      device="cuda", dtype=torch.bfloat16)
generator = LMGen(model)  # normal Moshi audio streaming interface
prompt_ids = vocabulary.encode("안녕하세요")
text = vocabulary.decode(generated_text_ids, helium_tokenizer,
                         special_ids={model.text_padding_token_id,
                                      model.end_of_text_padding_id})
```

Decode complete token runs, not individual BPE pieces: a Korean UTF-8 character
may span multiple tokens. An interactive UI needs byte buffering for partial
runs. The original Helium tokenizer is not modified.

The stock Moshi loader/server alone is NOT sufficient: it would change the
sentinel ID and cannot decode added tokens. Use this loader and mapping-aware
renderer. Quantized/Candle/MLX export, demultiplexed text streams, automatic
checkpoint training resume, and stock web UI integration are not implemented.
The official trainer does not restore optimizer/scheduler state; these changes
do not add exact training resume. The inference helper currently loads the base
model even for merged checkpoints and therefore needs access to it.

Multi-GPU FSDP follows the official meta/rank-zero initialization and checkpoint
path but has not been exercised on hardware. Tests run against installed Moshi
and PyTorch 2.9.0; the repository pins PyTorch 2.6, which was not available in
the selected environment. Dependencies on Whisper attention internals require
revalidation when Whisper changes.

## Validation

```bash
python -m pytest tests/test_korean.py -q
```

Tests use small instances of the actual official LMModel, real Whisper attention
and DTW with random test weights, and real safetensors serialization. They cover
original rows, both embedding gradients and output gradients, LoRA optimization,
consolidation, inference reload, autoregressive expanded-token generation,
English regression, stereo stream shapes, timestamp offsets, collisions and
explicit overflow. Random Whisper weights validate extraction mechanics only,
not alignment accuracy. A small CPU forward/backward/AdamW step is included;
no pretrained 7B training or Korean speech quality validation has been run.

Validation result in this workspace: **12 tests passed** (CPU), including full
and LoRA optimizer smoke steps. CUDA was unavailable (`torch.cuda.is_available()`
was false). The example `example/moshi_whisper_korean.yaml` parses successfully.

A synthetic 4-second sidecar for `안녕하세요 123, OpenAI!` was also processed by
the executable corpus command: 8 vocabulary entries, 100% occurrence coverage,
2.0 main-speaker tokens/second, 0% collisions, 0% shifted tokens, 0% dropped tokens.
These are fixture statistics, **not measurements of the user's training corpus**.

## Automatic preparation from existing turn annotations

The configured `example/moshi_whisper_korean.yaml` now enables
`korean_auto_prepare: true`. Existing stereo WAVs and sidecars containing
`turns: [{speaker, text, start, end}]` can be used directly. Rank zero scans
transcripts and creates `korean_vocab` before allocating the expanded model.
An existing vocabulary is validated and never silently renumbered. The example
sets `korean_vocab_include_eval: true`: evaluation token inventory contributes
to vocabulary coverage only; evaluation examples are not optimizer inputs.
Set it false for a strict train-only vocabulary and audit held-out coverage.

On a cache miss, the data loader aligns all selected turns in the original
recording with Whisper cross-attention/DTW, writes an atomic sidecar under
`korean_alignment_cache`, and reuses it across epochs, segments and restarts.
Source audio and annotation files are untouched. Cache keys include transcript,
audio path/size/mtime, channel, model name and language count. File locks prevent
concurrent ranks from generating the same recording twice. Invalidate the cache
if the implementation or weights behind the same model name are changed.

Defaults select `ASSISTANT` turns on channel 0, emitted as `SPEAKER_MAIN`.
Turns longer than 30 seconds or exceeding Whisper text context fail explicitly;
the loader never estimates token timing by dividing existing word timestamps.
Existing `tokens` sidecars are consumed directly. English behavior is unchanged
when automatic preparation is disabled. `whisper_device: cpu` keeps alignment
weights off the training GPU; the first pass is slower. Set it to `cuda` only
with enough memory for Whisper alongside Moshi. The lazy Whisper model stays
resident after first use. Original recording timestamps are preserved.

Workspace validation: 19 tests passed. The current 1,000 train/eval records
produced 2,392 distinct BPE IDs, with 100% selected transcript token coverage.
A real Whisper-small CPU alignment of the first assistant turn in daily-00686
produced 28 tokens in 4.11 seconds; cached reuse took 0.0002 seconds. Full corpus
alignment/collision statistics remain pending because alignment is generated
on demand. These timings are a single-window smoke test, not a throughput promise.
