"""Validation de la réponse backend ASR (§84-§88) — défense en profondeur.

Le backend applique déjà bornes et invariants avant `completed` ; la
Gateway RE-VALIDE la forme canonique et les invariants de timestamps à la
lecture. Toute divergence => 502 transcription_failed (jamais un résultat
partiel ni « réparé »). Aucune probabilité par segment/mot n'est admise.
"""

import re
from typing import Any, Dict, List

WORD_SLACK_MS = 200
END_SLACK_MS = 2000
MAX_SEGMENTS = 20000
MAX_WORDS = 200000
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_RESULT_JSON_BYTES = 32 * 1024 * 1024

_TOP_REQUIRED = ("transcription_id", "model_logical", "backend", "media", "duration_ms", "language_mode",
                 "language", "language_probability", "timestamps", "diarized", "speaker_count", "text",
                 "segments", "usage", "warnings")
_MEDIA_REQUIRED = ("container", "codec", "sample_rate", "channels", "upload_bytes", "sha256")
_USAGE_REQUIRED = ("audio_ms", "processing_ms", "rtf", "segment_count", "word_count", "vocabulary_count")
_FORBIDDEN_KEYS = ("probability", "avg_logprob", "no_speech_prob", "compression_ratio", "path", "filename")
# Diarisation V1 : étiquettes ANONYMES par job (speaker_00…), au plus 4
# locuteurs (Sortformer 4spk-v2) ; jamais de nom, jamais de locuteur par mot.
_SPEAKER_RE = re.compile(r"^speaker_\d{2}$")
MAX_SPEAKERS = 4


class ResponseInvalid(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_result(result: Any, expected_id: str, expected_model: str, expected_timestamps: str,
                    expected_diarize: bool = False) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ResponseInvalid("result_not_object")
    for key in _TOP_REQUIRED:
        if key not in result:
            raise ResponseInvalid(f"missing_{key}")
    if result["transcription_id"] != expected_id:
        raise ResponseInvalid("transcription_id_mismatch")
    if result["model_logical"] != expected_model:
        raise ResponseInvalid("model_mismatch")
    if result["timestamps"] != expected_timestamps or expected_timestamps not in ("segment", "word"):
        raise ResponseInvalid("timestamps_mismatch")
    diarized = result["diarized"]
    if diarized is not True and diarized is not False:
        raise ResponseInvalid("diarization_fields")
    if expected_diarize is False and diarized is not False:
        raise ResponseInvalid("diarization_fields")
    speaker_count = result["speaker_count"]
    if diarized:
        if not _int(speaker_count) or not 0 <= speaker_count <= MAX_SPEAKERS:
            raise ResponseInvalid("diarization_fields")
    elif speaker_count is not None:
        raise ResponseInvalid("diarization_fields")
    backend = result["backend"]
    if not isinstance(backend, dict) or set(backend) != {"provider", "artifact", "revision"}:
        raise ResponseInvalid("backend_shape")
    media = result["media"]
    if not isinstance(media, dict) or any(k not in media for k in _MEDIA_REQUIRED):
        raise ResponseInvalid("media_shape")
    if not _int(result["duration_ms"]) or result["duration_ms"] <= 0:
        raise ResponseInvalid("duration_invalid")
    if result["language_mode"] not in ("auto", "fr", "en") or not isinstance(result["language"], str):
        raise ResponseInvalid("language_invalid")
    prob = result["language_probability"]
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not 0.0 <= float(prob) <= 1.0:
        raise ResponseInvalid("language_probability_invalid")
    text = result["text"]
    if not isinstance(text, str):
        raise ResponseInvalid("text_invalid")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ResponseInvalid("output_bounds_exceeded")
    segments = result["segments"]
    if not isinstance(segments, list):
        raise ResponseInvalid("segments_invalid")
    counts = check_segments(segments, result["duration_ms"], expected_timestamps == "word", diarized=diarized)
    if diarized and counts["speakers_seen"] > speaker_count:
        raise ResponseInvalid("diarization_fields")
    if text != " ".join(s["text"] for s in segments):
        raise ResponseInvalid("text_segments_mismatch")
    usage = result["usage"]
    if not isinstance(usage, dict) or any(not _int(usage.get(k)) for k in _USAGE_REQUIRED if k != "rtf"):
        raise ResponseInvalid("usage_shape")
    if usage["segment_count"] != counts["segment_count"] or usage["word_count"] != counts["word_count"]:
        raise ResponseInvalid("usage_mismatch")
    warnings = result["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ResponseInvalid("warnings_invalid")
    _reject_forbidden_keys(result)
    return result


def _reject_forbidden_keys(node: Any, depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _FORBIDDEN_KEYS:
                raise ResponseInvalid("forbidden_field")
            if isinstance(v, (dict, list)) and k != "segments":
                _reject_forbidden_keys(v, depth + 1)
    if isinstance(node, list) and node:
        _reject_forbidden_keys(node[0], depth + 1)


def check_segments(segments: List[Any], duration_ms: int, words_expected: bool,
                   diarized: bool = False) -> Dict[str, int]:
    if len(segments) > MAX_SEGMENTS:
        raise ResponseInvalid("output_bounds_exceeded")
    words = 0
    speakers = set()
    prev_start, prev_end = -1, 0
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ResponseInvalid("segment_shape")
        if seg.get("segment_id") != f"seg_{i + 1:06d}":
            raise ResponseInvalid("segment_shape")
        speaker = seg.get("speaker")
        if speaker is not None:
            if not diarized or not isinstance(speaker, str) or not _SPEAKER_RE.match(speaker):
                raise ResponseInvalid("segment_shape")
            speakers.add(speaker)
            if len(speakers) > MAX_SPEAKERS:
                raise ResponseInvalid("diarization_fields")
        start, end = seg.get("start_ms"), seg.get("end_ms")
        if not _int(start) or not _int(end) or start < 0 or end <= start:
            raise ResponseInvalid("timestamp_invariant")
        # Monotonie : ordonnés, non décroissants, et sans chevauchement au-delà
        # de la tolérance (un chevauchement de quelques ms peut naître de
        # l'alignement des mots ; 1 s de recouvrement est une sortie invalide).
        if start < prev_start or end < prev_end or start < prev_end - WORD_SLACK_MS \
                or end > duration_ms + END_SLACK_MS:
            raise ResponseInvalid("timestamp_invariant")
        if not isinstance(seg.get("text"), str) or not seg["text"]:
            raise ResponseInvalid("segment_text")
        for k in seg:
            if k in _FORBIDDEN_KEYS:
                raise ResponseInvalid("forbidden_field")
        if words_expected != ("words" in seg):
            raise ResponseInvalid("words_presence")
        if "words" in seg:
            if not isinstance(seg["words"], list):
                raise ResponseInvalid("words_shape")
            words += len(seg["words"])
            if words > MAX_WORDS:
                raise ResponseInvalid("output_bounds_exceeded")
            wprev = start - WORD_SLACK_MS
            for w in seg["words"]:
                if not isinstance(w, dict) or set(w) != {"word", "start_ms", "end_ms"}:
                    raise ResponseInvalid("word_shape")
                ws, we = w["start_ms"], w["end_ms"]
                if not _int(ws) or not _int(we) or we <= ws or ws < start - WORD_SLACK_MS \
                        or we > end + WORD_SLACK_MS or ws < wprev:
                    raise ResponseInvalid("timestamp_invariant")
                if not isinstance(w["word"], str) or not w["word"]:
                    raise ResponseInvalid("word_shape")
                wprev = ws
        prev_start, prev_end = start, end
    return {"segment_count": len(segments), "word_count": words, "speakers_seen": len(speakers)}
