"""Admission d'image du rail vision (lot vision-1a).

Le décodage LOCAL fait foi (le MIME client est consultatif) : formats
statiques PNG/JPEG/WEBP uniquement, animation refusée, orientation EXIF
appliquée aux pixels, métadonnées EXIF (GPS, appareil…) JAMAIS transmises,
normalisation RGB avec aplatissement de la transparence sur fond blanc
neutre fixe (comportement déterministe documenté, pas un accident de lib).

Aucune image ni base64 n'apparaît dans un message d'erreur ou un log.
"""

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from .settings import settings

_FORMAT_BY_PIL = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}
_MIME_BY_FORMAT = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

# Identité VOLONTAIRE du normaliseur (lot CV-1). Elle nomme le contrat
# ci-dessus : décodage PNG/JPEG/WEBP, MIME == format décodé, animation
# refusée, EXIF transpose, borne de pixels, alpha aplati sur blanc, RGB,
# ré-encodage PNG sans métadonnées. Elle entre dans le fingerprint du
# pipeline. Tout changement qui modifie UN octet du PNG produit (code OU
# montée de Pillow constatée par les vecteurs de test
# tests/data/vision_normalizer) exige soit de conserver les octets, soit un
# bump explicite vers `/2` — jamais une version Pillow glissée dans l'identité.
NORMALIZER_ID = "vision-image-normalizer/1"
NORMALIZED_FORMAT = "png"


class ImageAdmissionError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass(frozen=True)
class AdmittedImage:
    """Image admise : diagnostics faisant foi + représentation normalisée."""

    width: int          # après orientation EXIF
    height: int
    pixels: int
    bytes: int          # taille de l'image SOUMISE (octets décodés du base64)
    format: str         # png|jpeg|webp (décodé, pas déclaré)
    # Provenance technique (lot CV-1) : SHA-256 des octets SOUMIS (décodés du
    # base64, avant Pillow, EXIF compris = identité de fichier) et SHA-256 des
    # octets EXACTS de `normalized_png` (= ce que le modèle voit).
    source_sha256: str
    normalized_sha256: str
    normalized_png: bytes   # RGB, sans métadonnées — ce que 1B enverra à Ollama
    normalized_mime: str = "image/png"


def _too_large(detail: str) -> ImageAdmissionError:
    return ImageAdmissionError("image_too_large", detail, http_status=413)


def admit_image(raw: bytes, declared_mime: str) -> AdmittedImage:
    if len(raw) > settings.vision_max_image_bytes:
        raise _too_large(
            f"image de {len(raw)} octets > borne admise "
            f"({settings.vision_max_image_bytes})"
        )

    # Garde anti-bombe de décompression : Pillow vérifie les pixels déclarés
    # dans l'en-tête AVANT allocation complète.
    Image.MAX_IMAGE_PIXELS = settings.vision_max_image_pixels

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Image.DecompressionBombError as exc:
        raise _too_large("image dépasse la borne de pixels admise") from exc
    except Exception as exc:  # noqa: BLE001 — octets non décodables
        raise ImageAdmissionError(
            "image_decode_failed", "octets non décodables en image"
        ) from exc

    fmt = _FORMAT_BY_PIL.get(img.format or "")
    if fmt is None:
        raise ImageAdmissionError(
            "unsupported_image_type",
            f"format décodé non supporté : {img.format or 'inconnu'}",
        )
    if _MIME_BY_FORMAT[fmt] != declared_mime:
        # Spoof MIME : rejeté (le décodage fait foi, on ne « corrige » pas).
        raise ImageAdmissionError(
            "unsupported_image_type",
            f"mime_type déclaré {declared_mime} mais format décodé {fmt}",
        )
    if getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) > 1:
        # V1 : média animé refusé — jamais d'analyse silencieuse de la frame 1.
        raise ImageAdmissionError(
            "unsupported_image_type", "média animé/multi-frames non supporté en V1"
        )

    # Orientation EXIF appliquée AUX PIXELS ; les diagnostics reflètent les
    # dimensions normalisées (contrat §19, documenté).
    img = ImageOps.exif_transpose(img)

    if img.width * img.height > settings.vision_max_image_pixels:
        raise _too_large(
            f"image de {img.width * img.height} pixels > borne admise "
            f"({settings.vision_max_image_pixels})"
        )

    # Normalisation couleur déterministe : alpha aplati sur fond BLANC fixe
    # (neutre, non configurable) ; niveaux de gris/palette convertis en RGB.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        normalized = background
    else:
        normalized = img.convert("RGB")

    # Ré-encodage PNG SANS métadonnées : c'est la seule représentation
    # transmise au backend — l'EXIF d'origine (GPS, appareil) meurt ici.
    out = io.BytesIO()
    normalized.save(out, format="PNG")
    normalized_png = out.getvalue()

    return AdmittedImage(
        width=normalized.width,
        height=normalized.height,
        pixels=normalized.width * normalized.height,
        bytes=len(raw),
        format=fmt,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        normalized_sha256=hashlib.sha256(normalized_png).hexdigest(),
        normalized_png=normalized_png,
    )
