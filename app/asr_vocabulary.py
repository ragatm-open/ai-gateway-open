"""Validation du vocabulaire (§39-§43) — copie IDENTIQUE de
asr/asr_backend/vocabulary.py (test de parité dans tests/test_asr_routes.py) :
la Gateway refuse AVANT tout octet, le backend re-valide. Aucun terme n'est
jamais journalisé : seule la cardinalité (`vocabulary_count`) sort d'ici.
"""

import re
import unicodedata
from typing import List, Tuple

MAX_ENTRIES = 64
MIN_ENTRY_CHARS = 2
MAX_ENTRY_CHARS = 48
MAX_TOTAL_CHARS = 1024
# Lettres (toute écriture), chiffres, espace, apostrophes, tiret, point, &.
_ALLOWED_RE = re.compile(r"^[\w \'’\-\.&]+$", re.UNICODE)


class VocabularyError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def normalize_vocabulary(raw: object) -> Tuple[List[str], int]:
    """Retourne (termes normalisés dédupliqués, cardinalité). Lève
    VocabularyError avec une raison NON contextuelle (jamais le terme)."""
    if raw is None:
        return [], 0
    if not isinstance(raw, list):
        raise VocabularyError("vocabulary doit être une liste de chaînes")
    if len(raw) == 0:
        raise VocabularyError("vocabulary ne peut pas être vide")
    if len(raw) > MAX_ENTRIES:
        raise VocabularyError(f"vocabulary : au plus {MAX_ENTRIES} entrées")
    out: List[str] = []
    seen = set()
    total = 0
    for item in raw:
        if not isinstance(item, str):
            raise VocabularyError("vocabulary : chaque entrée doit être une chaîne")
        term = unicodedata.normalize("NFC", item).strip()
        if not term:
            raise VocabularyError("vocabulary : entrée vide")
        if "\n" in term or "\r" in term or any(ord(c) < 32 or ord(c) == 127 for c in term):
            raise VocabularyError("vocabulary : caractère de contrôle interdit")
        if "://" in term:
            raise VocabularyError("vocabulary : URL interdite")
        if not MIN_ENTRY_CHARS <= len(term) <= MAX_ENTRY_CHARS:
            raise VocabularyError(f"vocabulary : chaque entrée fait {MIN_ENTRY_CHARS}..{MAX_ENTRY_CHARS} caractères")
        if not _ALLOWED_RE.match(term) or "_" in term:
            raise VocabularyError("vocabulary : caractères admis = lettres, chiffres, espace, ' - . &")
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        total += len(term)
        if total > MAX_TOTAL_CHARS:
            raise VocabularyError(f"vocabulary : au plus {MAX_TOTAL_CHARS} caractères au total")
        out.append(term)
    return out, len(out)


def hotwords_string(terms: List[str]) -> str:
    """Forme relayée à faster-whisper (`hotwords`, jamais `initial_prompt`)."""
    return ", ".join(terms)
