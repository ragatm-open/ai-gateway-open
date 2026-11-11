#!/usr/bin/env sh
# SBOM de l'image Gateway (un lot antérieur, toolchain épinglée d'une version antérieure)
# artefact de BUILD produit côté hôte, rien n'est ajouté au runtime de
# l'image. Syft s'exécute dans un conteneur éphémère (image épinglée par tag
# ET digest dans tools/toolchain.env) sur l'export `docker save` de l'image :
# aucun accès au socket Docker, aucun réseau une fois l'outil présent.
# Usage : tools/sbom.sh <image:tag> [répertoire de sortie]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/tools/toolchain.env"
IMG="${1:?image requise}"; OUT="${2:-.}"
NAME=$(echo "$IMG" | tr '/:' '__')
CDX="$OUT/sbom-$NAME.cdx.json"; SPDX="$OUT/sbom-$NAME.spdx.json"

V=$(docker run --rm --network none "$SYFT_IMAGE" version 2>/dev/null | awk '/^Version:/{print $2}')
[ "$V" = "$SYFT_VERSION" ] || { echo "ERREUR : syft $V != $SYFT_VERSION attendu ($SYFT_IMAGE)"; exit 1; }

TAR=$(mktemp /tmp/sbom-XXXXXX.tar)
trap 'rm -f "$TAR"' EXIT
docker save "$IMG" -o "$TAR"
docker run --rm --network none -v "$TAR:/image.tar:ro" -v "$OUT:/out" "$SYFT_IMAGE" \
  scan "docker-archive:/image.tar" \
  -o "cyclonedx-json=/out/$(basename "$CDX")" \
  -o "spdx-json=/out/$(basename "$SPDX")" -q

python3 - "$CDX" "syft $V ($SYFT_IMAGE)" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1])); comps = d.get("components", [])
eco = Counter((c.get("purl") or "pkg:?/").split(":")[1].split("/")[0] for c in comps)
print(f"SBOM {d.get('bomFormat')} {d.get('specVersion')} via {sys.argv[2]} : {len(comps)} composants {dict(eco)} -> {sys.argv[1]}")
PY
echo "SPDX : $SPDX"
