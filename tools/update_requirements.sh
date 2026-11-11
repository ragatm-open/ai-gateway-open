#!/usr/bin/env sh
# Régénération REPRODUCTIBLE des locks.
#
# Même image Python que le runtime (digest lu dans le FROM du Dockerfile),
# pip-tools épinglé (tools/toolchain.env), mêmes options qu'une version antérieure.
# Les fichiers du dépôt ne sont remplacés QUE si les deux générations ont
# réussi ; en cas d'erreur, rien n'est modifié.
#
# Usage : tools/update_requirements.sh        (depuis la racine du dépôt)
#   1. modifier requirements.in / requirements-dev.in (jamais les .txt)
#   2. lancer ce script, puis `git diff` : seules les lignes voulues bougent
# Sans modification des .in, la régénération doit rendre un diff VIDE —
# pip-compile conserve les pins existants tant qu'on ne passe pas --upgrade.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
. "$ROOT/tools/toolchain.env"

# L'image de génération = l'image runtime, sans dupliquer le digest.
LOCK_IMAGE=$(grep -m1 '^FROM ' Dockerfile | awk '{print $2}')
case "$LOCK_IMAGE" in
  python:3.12-slim@sha256:*) ;;
  *) echo "ERREUR : FROM du Dockerfile non épinglé par digest ($LOCK_IMAGE)"; exit 1 ;;
esac

WORK=$(mktemp -d /tmp/lock-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
# Les locks existants sont fournis à pip-compile pour qu'il CONSERVE les pins.
cp requirements.in requirements-dev.in requirements.txt requirements-dev.txt "$WORK/"

docker run --rm -v "$WORK:/w" -w /w "$LOCK_IMAGE" sh -c "
  set -e
  pip install -q --no-cache-dir 'pip-tools==$PIP_TOOLS_VERSION'
  V=\$(pip-compile --version | awk '{print \$3}')
  [ \"\$V\" = '$PIP_TOOLS_VERSION' ] || { echo \"ERREUR : pip-compile \$V != $PIP_TOOLS_VERSION\"; exit 1; }
  echo \"pip-compile \$V dans $LOCK_IMAGE\"
  pip-compile --quiet --generate-hashes --no-strip-extras --allow-unsafe -o requirements.txt requirements.in
  pip-compile --quiet --generate-hashes --no-strip-extras -o requirements-dev.txt requirements-dev.in
"
# Succès des deux générations : remplacement atomique des fichiers du dépôt.
cp "$WORK/requirements.txt" requirements.txt
cp "$WORK/requirements-dev.txt" requirements-dev.txt
echo "locks régénérés — vérifier : git diff --stat requirements.txt requirements-dev.txt"
