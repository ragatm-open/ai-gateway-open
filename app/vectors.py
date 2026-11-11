"""Contrôles et normalisation des vecteurs d'embedding.

Ce module ne parle ni HTTP ni Ollama : il valide des listes de flottants. La
Gateway l'utilise pour refuser des vecteurs inexploitables *avant* qu'ils
n'atteignent un index vectoriel, où ils seraient bien plus coûteux à détecter.

Cas réellement observés sur l'installation d'origine et couverts ici :

* `/api/embeddings` (chemin de repli historique) renvoie des vecteurs **non
  normalisés** — ‖v‖ ≈ 19,7 pour nomic-embed-text — alors que `/api/embed`
  renvoie ‖v‖ = 1. Mélanger les deux dans une même colonne pgvector est muet
  avec l'opérateur cosinus `<=>` et faux avec le produit scalaire `<#>`.
* `json.loads` accepte `NaN` et `Infinity` : un vecteur non fini peut donc
  traverser tout le pipeline sans erreur et corrompre un index HNSW.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

# Tolérance admise autour de 1,0 pour un profil déclaré `normalized: true`.
# Mesuré sur les trois modèles installés : écart-type des normes ≤ 2,6e-07.
# La fourchette retenue laisse largement la place au bruit de sérialisation
# JSON tout en détectant un vecteur non normalisé (‖v‖ ≈ 19 à 26).
NORM_MIN = 0.99
NORM_MAX = 1.01


class VectorError(ValueError):
    """Vecteur inexploitable. Le message ne contient jamais de texte métier."""


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in vector))


def is_finite(vector: Iterable[float]) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)


def l2_normalize(vector: Sequence[float]) -> List[float]:
    """Normalise en L2. Lève si la norme est nulle ou non finie."""
    norm = l2_norm(vector)
    if not math.isfinite(norm) or norm <= 0.0:
        raise VectorError("norme L2 nulle ou non finie : vecteur non normalisable")
    return [float(x) / norm for x in vector]


def norm_in_tolerance(norm: float) -> bool:
    return math.isfinite(norm) and NORM_MIN <= norm <= NORM_MAX


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimension: int | None = None,
    require_normalized: bool = False,
) -> int:
    """Valide un lot de vecteurs et retourne la dimension observée.

    Contrôles, dans l'ordre où un opérateur voudrait les lire :
      cardinalité · vecteur non vide · dimension attendue · homogénéité ·
      valeurs finies · vecteur non nul · norme dans la tolérance.

    Lève :
        VectorError avec un message qui cite l'index fautif, jamais le texte.
    """
    if len(vectors) != expected_count:
        raise VectorError(
            f"nombre de vecteurs ({len(vectors)}) != nombre de textes ({expected_count})"
        )
    if expected_count == 0:
        raise VectorError("aucun vecteur à valider")

    first = vectors[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        raise VectorError("vecteur 0 vide")

    dimension = len(first)
    if expected_dimension is not None and dimension != expected_dimension:
        raise VectorError(
            f"dimension inattendue ({dimension} != {expected_dimension})"
        )

    for index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)) or len(vector) == 0:
            raise VectorError(f"vecteur {index} vide")
        if len(vector) != dimension:
            raise VectorError(
                f"vecteur {index} de dimension {len(vector)} != {dimension}"
            )
        if not is_finite(vector):
            raise VectorError(
                f"vecteur {index} contient une valeur non finie (NaN ou Infinity)"
            )
        norm = l2_norm(vector)
        if norm <= 0.0:
            raise VectorError(f"vecteur {index} de norme nulle")
        if require_normalized and not norm_in_tolerance(norm):
            raise VectorError(
                f"vecteur {index} de norme {norm:.6f} hors tolérance "
                f"[{NORM_MIN}, {NORM_MAX}] pour un modèle déclaré normalisé"
            )

    return dimension
