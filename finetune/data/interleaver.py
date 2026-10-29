import json
import math
import os
import warnings
from collections import deque
from dataclasses import dataclass
from functools import reduce

import numpy as np
import sentencepiece
import torch
from moshi.conditioners import ConditionAttributes

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        if batch[0].condition_attributes is None:
            return Batch(codes)
        return Batch(codes, [b.condition_attributes for b in batch])


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


class Interleaver:
    """Interleaver with basic featuress
    Args:
        tokenizer: text tokenizer used by the model.
        audio_frame_rate (float): frame rate of the audio tokenizer.
        text_padding (int): special token used for text padding.
        end_of_text_padding (int): special token used to indicate end of text padding.
        zero_padding (int): special token id indicating that a 0 should be used instead
            of an actual embedding.
        in_word_padding (int | None): padding used within a word segment. Will default to `text_padding`.
        keep_main_only (bool): if True, will only keep the alignments with the main speaker.
        keep_and_shift (bool): if True, will not drop any alignment, except for those with negative duration.
        use_bos_eos: (bool): if True, inserts BOS, EOS for change of turns.
        audio_delay (float): delay between the text and audio.
            A positive value means the text will be ahead of the audio.
        proba (float): probability of keeping the text.
        device: device location for the output tensors.
    """

    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
        korean_vocabulary=None,
        korean_overflow: str = "error",
    ):
        self.korean_vocabulary = korean_vocabulary
        if korean_overflow not in {"error", "warn"}:
            raise ValueError("korean_overflow must be error or warn")
        self.korean_overflow = korean_overflow
        self.korean_stats = {}
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    @property
    def special_tokens(self) -> set[int]:
        """Return the set of special tokens used by this interleaver."""
        return {
            self.text_padding,
            self.end_of_text_padding,
            self.tokenizer.bos_id(),
            self.tokenizer.eos_id(),
            self.zero_padding,
            self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        # Tokenizes each word individually into a list of ints.
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Removes all words with negative or 0 durations.
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Delay the audio with respect to the text, e.g. positive values mean the audio is late on the text.
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        # EOS and BOS is different from what it was in the old Interleaver, it is now symmetrical:
        # if the main speaker talks after another speaker (or is the first to talk), BOS is prepended to the first word.
        # Similary, if any other speaker speaks either first, or after the main speaker, a EOS is prepended.
        # This is in contrast with the legacy Interleaver, where the EOS would be inserted immediately
        # at the end of the turn of the main speaker.
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Builds the token stream from the tokenized alignments."""
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_korean(self, tokens, start_sec, num_frames, main_speaker=None):
        """Absolute timestamps, half-open segment ownership by token start.

        Stable start-time ordering resolves overlap. Queue overflow is fatal by
        default; warn mode exposes counts and an explicit warning.
        """
        if self.korean_vocabulary is None:
            raise ValueError("Korean token records require korean_vocab")
        if self.audio_frame_rate != 12.5:
            raise ValueError("Korean alignment requires Mimi at 12.5 Hz")
        if self.audio_delay != 0 or self.use_bos_eos:
            raise ValueError("Korean path currently requires zero delay and no turn BOS/EOS")
        mapping = self.korean_vocabulary.mapping
        stats = dict(tokens=0, collisions=0, shifted=0, dropped=0, filtered=0)
        stream = [self.text_padding] * num_frames
        next_free, occupied = 0, set()
        end = start_sec + num_frames / self.audio_frame_rate
        for token in sorted(tokens, key=lambda t: t["start"]):
            start, stop = float(token["start"]), float(token["end"])
            if not math.isfinite(start) or not math.isfinite(stop) or start < 0 or stop < start:
                raise ValueError(f"Invalid token timestamp: {token}")
            placement = token.get("moshi_frame")
            owner_time = start if placement is None else placement / self.audio_frame_rate
            if not start_sec <= owner_time < end:
                continue
            if self.keep_main_only and token.get("speaker", self.main_speaker_label) != (main_speaker or self.main_speaker_label):
                stats["filtered"] += 1
                continue
            token_id = mapping[token["whisper_token_id"]]  # unknown IDs fail loudly
            frame = min(num_frames - 1, math.floor((owner_time - start_sec) * self.audio_frame_rate + 1e-8))
            stats["tokens"] += 1
            stats["collisions"] += int(token.get("frame_collision", frame in occupied))
            occupied.add(frame)
            target = max(frame, next_free)
            stats["shifted"] += int(token.get("frame_shifted", target != frame))
            if target >= num_frames:
                stats["dropped"] += 1
                continue
            if target > 0 and stream[target - 1] in {self.text_padding, self.in_word_padding}:
                stream[target - 1] = self.end_of_text_padding
            stream[target] = token_id
            next_free = target + 1
            stop_frame = min(num_frames, math.floor((stop - start_sec) * self.audio_frame_rate) + 1)
            for index in range(next_free, stop_frame):
                stream[index] = self.in_word_padding
        self.korean_stats = stats
        if stats["dropped"]:
            message = f"Korean frame capacity exceeded: {stats} (segment start={start_sec})"
            if self.korean_overflow == "error":
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
        return torch.tensor(stream, device=self.device).view(1, 1, -1)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        """Responsible with processing the alignments and calling `build_token_stream`."""
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


class InterleavedTokenizer:
    def __init__(self, mimi, interleaver, duration_sec: float, alignment_cache=None):
        self.alignment_cache = alignment_cache
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        if interleaver.korean_vocabulary is not None and not math.isclose(duration_sec * mimi.frame_rate, round(duration_sec * mimi.frame_rate)):
            raise ValueError("Korean segment duration must be a multiple of 80ms")
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> Sample:
        with torch.no_grad():
            if wav.ndim != 2 or wav.shape[0] != 2:
                raise ValueError("Expected stereo audio with original channel order")
            audio_tensor = torch.as_tensor(wav, device=self.interleaver.device)
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
            audio_tokens = audio_tokens[..., : self.num_audio_frames]
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens[..., : self.num_audio_frames],
                (0, self.num_audio_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

            info_file = os.path.splitext(path)[0] + ".json"
            with open(info_file) as f:
                data = json.load(f)
            if self.alignment_cache is not None:
                data = self.alignment_cache(path, data)
            if "tokens" in data:
                vocabulary = self.interleaver.korean_vocabulary
                if vocabulary is None or data.get("num_languages", vocabulary.num_languages) != vocabulary.num_languages:
                    raise ValueError("Missing or incompatible Korean vocabulary")
                text_tokens = self.interleaver.prepare_korean(
                    data["tokens"], start_sec, this_num_audio_frames
                )
            else:
                alignments = data["alignments"]
                start_alignment = dicho(alignments, start_sec)
                end_alignment = dicho(alignments, start_sec + self.duration_sec)
                alignments = [
                    (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                    for a in alignments[start_alignment:end_alignment]
                ]
                text_tokens = self.interleaver.prepare_item(
                    alignments, this_num_audio_frames / self.mimi.frame_rate
                )
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )

            if text_tokens.shape[-1] != audio_tokens.shape[-1]:
                raise ValueError("Text/audio frame count mismatch")
            codes = torch.cat([text_tokens, audio_tokens], dim=1)
            return Sample(codes, data.get("text_conditions", None))
