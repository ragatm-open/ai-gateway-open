"""Projections du résultat canonique (§89-§92) : text, srt, vtt.

Le JSON reste la seule forme AUTORITATIVE ; les projections dérivent des
segments (jamais des mots) et n'ajoutent aucune information.
"""

from typing import Any, Dict


def _stamp(ms: int, sep: str) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{milli:03d}"


def to_text(result: Dict[str, Any]) -> str:
    return result["text"] + "\n"


def to_srt(result: Dict[str, Any]) -> str:
    lines = []
    for i, seg in enumerate(result["segments"], start=1):
        lines.append(str(i))
        lines.append(f"{_stamp(seg['start_ms'], ',')} --> {_stamp(seg['end_ms'], ',')}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def to_vtt(result: Dict[str, Any]) -> str:
    lines = ["WEBVTT", ""]
    for seg in result["segments"]:
        lines.append(f"{_stamp(seg['start_ms'], '.')} --> {_stamp(seg['end_ms'], '.')}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines) + "\n"


MEDIA_TYPES = {"json": "application/json", "text": "text/plain; charset=utf-8",
               "srt": "application/x-subrip; charset=utf-8", "vtt": "text/vtt; charset=utf-8"}
PROJECTIONS = {"text": to_text, "srt": to_srt, "vtt": to_vtt}
