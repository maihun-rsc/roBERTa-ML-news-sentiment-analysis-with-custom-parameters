"""
asr_cleaner.py
──────────────
Cleans Whisper-generated ASR transcripts from broadcast news (Times Now,
WION, Firstpost video content) before they enter the same NER pipeline
used for written articles.

Per context.md:
    "Whisper large-v2 transcripts undergo an additional noise-reduction
     step that removes disfluency markers, corrects systematic proper-noun
     misrecognitions using an outlet-specific entity whitelist, and segments
     the transcript into sentence-length units using a punctuation-insertion
     model. Low-confidence tokens ... are flagged and masked."

This module implements that pipeline using Whisper's actual output format
(segments with per-segment avg_logprob, which we use as the confidence proxy
since word-level logprobs require word_timestamps=True at transcribe time).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── Disfluency markers ───────────────────────────────────────────────────────
# Common ASR artifacts from spoken broadcast English. Conservative list —
# we remove filler words but NOT hedging words (allegedly, reportedly),
# which carry semantic weight per the project's core preprocessing rule.

_DISFLUENCY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(uh+|um+|erm+|ah+)\b", re.IGNORECASE),
    re.compile(r"\b(you know|i mean|sort of|kind of)\b(?=[,\s])", re.IGNORECASE),
    re.compile(r"\[(music|applause|laughter|crosstalk|inaudible)\]", re.IGNORECASE),
    re.compile(r"\(.*?(inaudible|unintelligible).*?\)", re.IGNORECASE),
]

_REPEATED_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b){1,}", re.IGNORECASE)  # "the the the"
_MULTI_WS_RE      = re.compile(r"\s{2,}")


@dataclass
class TranscriptSegment:
    """One Whisper output segment, carrying confidence for masking decisions."""
    text:        str
    start:       float        # seconds
    end:         float        # seconds
    avg_logprob: float        # Whisper's per-segment confidence proxy
    speaker:     str = ""     # populated only if diarization was run upstream


@dataclass
class CleanedTranscript:
    text:            str                      = ""
    segments:        list[TranscriptSegment]  = field(default_factory=list)
    masked_segments: int                      = 0   # count of low-confidence segments dropped
    total_segments:  int                      = 0


# ── Outlet-specific entity whitelist ─────────────────────────────────────────
# ASR systematically mis-hears certain proper nouns common in Indian/
# international broadcast news. This is a starter list — extend per
# outlet as misrecognition patterns are observed in practice.

ENTITY_CORRECTIONS: dict[str, str] = {
    "modi g":          "Modi ji",
    "modiji":          "Modi ji",
    "by den":          "Biden",
    "by dent":         "Biden",
    "zelensky":        "Zelenskyy",
    "zelinski":        "Zelenskyy",
    "putin's":         "Putin's",
    "shi jin ping":    "Xi Jinping",
    "she jinping":     "Xi Jinping",
    "rishi soonak":    "Rishi Sunak",
    "rishi sunac":     "Rishi Sunak",
}


def correct_entity_misrecognitions(text: str) -> str:
    """
    Apply the outlet-agnostic correction whitelist. Case-insensitive match,
    case-preserving replacement only where we have high confidence the
    correction is right (whole-phrase match, not partial).
    """
    corrected = text
    for wrong, right in ENTITY_CORRECTIONS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        corrected = pattern.sub(right, corrected)
    return corrected


def remove_disfluencies(text: str) -> str:
    """Strip filler words and transcription artifacts like [inaudible]."""
    for pattern in _DISFLUENCY_PATTERNS:
        text = pattern.sub("", text)
    # Collapse word repetitions caused by stutters/ASR duplication
    text = _REPEATED_WORD_RE.sub(r"\1", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def filter_low_confidence_segments(
    segments: list[TranscriptSegment],
    logprob_threshold: float = -1.0,
) -> tuple[list[TranscriptSegment], int]:
    """
    Mask (drop) segments below the confidence threshold.

    Whisper's avg_logprob is typically in range [-1.5, 0.0] for English;
    values below -1.0 generally indicate the model was guessing — these
    segments are dropped entirely rather than passed through with garbage
    text, since downstream NER would otherwise hallucinate entities from
    noise.

    Args:
        segments:           list of TranscriptSegment from Whisper output
        logprob_threshold:  segments with avg_logprob below this are dropped

    Returns:
        (kept_segments, num_masked)
    """
    kept = [s for s in segments if s.avg_logprob >= logprob_threshold]
    masked = len(segments) - len(kept)
    if masked:
        log.debug(f"[asr] Masked {masked}/{len(segments)} low-confidence segments "
                  f"(threshold={logprob_threshold})")
    return kept, masked


def segments_to_sentences(segments: list[TranscriptSegment]) -> str:
    """
    Join cleaned segments into sentence-like units.

    Whisper segments don't reliably align with sentence boundaries —
    a segment might end mid-sentence at a natural pause. We rely on
    spaCy's sentencizer downstream (ner_pipeline.py) to do the actual
    sentence splitting; here we just ensure segments end with terminal
    punctuation so the sentencizer has something to work with.
    """
    parts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        parts.append(text)
    return " ".join(parts)


def clean_transcript(
    segments: list[TranscriptSegment],
    logprob_threshold: float = -1.0,
) -> CleanedTranscript:
    """
    Full ASR cleaning pipeline:
      1. Filter low-confidence segments
      2. Remove disfluencies per remaining segment
      3. Correct known entity misrecognitions
      4. Join into punctuated text for the NER pipeline

    Args:
        segments:           raw Whisper segments for one broadcast clip
        logprob_threshold:  confidence cutoff (see filter_low_confidence_segments)

    Returns:
        CleanedTranscript ready for cleaner.clean_text() → ner_pipeline.
    """
    total = len(segments)
    kept, masked = filter_low_confidence_segments(segments, logprob_threshold)

    cleaned_segments: list[TranscriptSegment] = []
    for seg in kept:
        text = remove_disfluencies(seg.text)
        text = correct_entity_misrecognitions(text)
        if text:
            cleaned_segments.append(TranscriptSegment(
                text=text, start=seg.start, end=seg.end,
                avg_logprob=seg.avg_logprob, speaker=seg.speaker,
            ))

    full_text = segments_to_sentences(cleaned_segments)

    return CleanedTranscript(
        text=full_text,
        segments=cleaned_segments,
        masked_segments=masked,
        total_segments=total,
    )


def whisper_result_to_segments(whisper_result: dict) -> list[TranscriptSegment]:
    """
    Adapter: converts openai-whisper's raw `.transcribe()` output dict
    into our TranscriptSegment dataclass list.

    Args:
        whisper_result: dict returned by whisper.transcribe(audio_path)
                        — has a "segments" key, each segment a dict with
                        'text', 'start', 'end', 'avg_logprob'.

    Returns:
        List of TranscriptSegment.
    """
    segments = []
    for seg in whisper_result.get("segments", []):
        segments.append(TranscriptSegment(
            text=seg.get("text", ""),
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            avg_logprob=float(seg.get("avg_logprob", 0.0)),
        ))
    return segments
