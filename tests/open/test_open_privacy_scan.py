"""Conformité Open — scan de confidentialité : aucune trace privée dans
l'arbre candidat (hôtes, adresses, chemins personnels, identités, projets
voisins, infrastructure d'exploitation) — motifs génériques ; les identifiants nominatifs
relèvent de l'outillage interne non publié."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELF = os.path.abspath(__file__)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}

# Motifs GÉNÉRIQUES uniquement (adresses privées, domaines locaux, chemins personnels, infrastructure
# d'exploitation, formats de secrets). Les identifiants nominatifs (personnes, comptes, clients, hôtes)
# sont contrôlés par l'outillage interne du programme, qui n'est pas publié : une liste de noms privés
# dans un test public serait elle-même une fuite.
PATTERNS = [
    ("private-ip", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
    ("private-ip-10", re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("lan-domain", re.compile(r"\b[a-z0-9-]+\.lan\b", re.I)),
    ("home-path", re.compile(r"/home/[a-z][a-z0-9_-]*")),
    # chaînes assemblées : ce fichier ne doit pas se détecter lui-même (motif générique, non nominatif)
    ("prod-infra", re.compile("ragatm" + r"[_-](prod|staging)|/opt/" + r"apps/|/opt/" + r"ai/", re.I)),
    ("personal-mail", re.compile(r"[A-Za-z0-9._%+-]+@(gmail|hotmail|outlook|yahoo|free|orange|wanadoo)\.(com|fr)", re.I)),
    ("private-url", re.compile(r"https?://(192\.168\.|10\.|[a-z0-9-]+\.lan)", re.I)),
    ("secret-openai", re.compile(r"\bsk-(proj-)?[A-Za-z0-9_-]{20,}")),
    ("secret-aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret-pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def _files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            path = os.path.join(base, f)
            if path == SELF:
                continue
            yield path


def test_no_private_trace_anywhere():
    hits = []
    for path in _files():
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for pid, rx in PATTERNS:
            if rx.search(text):
                hits.append((os.path.relpath(path, ROOT), pid))
    assert hits == [], hits
