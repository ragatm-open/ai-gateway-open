"""Régénération des vecteurs du normaliseur d'image (fixtures SYNTHÉTIQUES).

Les sept images sont des formes Pillow (rectangle, ellipse, trait) : aucune
donnée réelle, aucune métadonnée personnelle. Les attendus (`vectors.json`)
sont les octets EXACTS du PNG normalisé par `app.vision_image.admit_image` au
moment de la génération : ils figent le comportement du normaliseur
(`NORMALIZER_ID`) contre toute dérive de code ou de version Pillow.

Usage (racine du dépôt) : python tests/data/vision_normalizer/generate.py
Toute régénération DOIT s'accompagner d'une revue : si les octets changent
sans changement voulu du normaliseur, c'est une régression, pas une mise à jour.
"""
import hashlib
import io
import json
import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import PIL  # noqa: E402

from app.vision_image import NORMALIZER_ID, admit_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def shapes(mode, size=(96, 64), bg=(240, 240, 240)):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 40, 40], fill=(200, 30, 30))
    d.ellipse([50, 12, 88, 52], fill=(30, 120, 200))
    d.line([0, 60, 96, 4], fill=(20, 160, 60), width=3)
    return img.convert(mode)


def save(name, img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    data = buf.getvalue()
    with open(os.path.join(HERE, name), "wb") as fh:
        fh.write(data)
    return data


def exif_with_orientation(orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


def main():
    files = {
        "jpeg_rgb.jpeg": ("jpeg", "image/jpeg", save("jpeg_rgb.jpeg", shapes("RGB"), "JPEG", quality=90)),
        "jpeg_exif1.jpeg": ("jpeg", "image/jpeg", save("jpeg_exif1.jpeg", shapes("RGB"), "JPEG", quality=90,
                                                       exif=exif_with_orientation(1))),
        # Orientation 6 : mêmes pixels stockés 96x64, tag EXIF 6 => le normaliseur
        # transpose et rend 64x96 (le test vérifie la transposition).
        "jpeg_exif6.jpeg": ("jpeg", "image/jpeg", save("jpeg_exif6.jpeg", shapes("RGB"), "JPEG", quality=90,
                                                       exif=exif_with_orientation(6))),
        "png_gray.png": ("png", "image/png", save("png_gray.png", shapes("L"), "PNG")),
        "png_rgb.png": ("png", "image/png", save("png_rgb.png", shapes("RGB"), "PNG")),
    }
    rgba = shapes("RGBA")
    rgba.putalpha(128)
    files["png_rgba.png"] = ("png", "image/png", save("png_rgba.png", rgba, "PNG"))
    files["webp_rgb.webp"] = ("webp", "image/webp", save("webp_rgb.webp", shapes("RGB"), "WEBP", quality=90))

    vectors = {}
    for fname, (fmt, mime, data) in files.items():
        admitted = admit_image(data, mime)
        assert admitted.format == fmt
        vectors[fname.rsplit(".", 1)[0]] = {
            "file": fname, "input_format": fmt, "mime_type": mime,
            "width": admitted.width, "height": admitted.height,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "normalized_bytes": len(admitted.normalized_png),
            "normalized_sha256": admitted.normalized_sha256,
        }
    out = {
        "generated_with_pillow": PIL.__version__,
        "normalizer_id": NORMALIZER_ID,
        "note": "images synthétiques (formes Pillow) régénérées via "
                "tests/data/vision_normalizer/generate.py ; aucune donnée réelle ; "
                "attendus = octets exacts du PNG normalisé",
        "vectors": dict(sorted(vectors.items())),
    }
    with open(os.path.join(HERE, "vectors.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"{len(vectors)} vecteurs régénérés ({NORMALIZER_ID}, Pillow {PIL.__version__})")


if __name__ == "__main__":
    main()
