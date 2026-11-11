#!/usr/bin/env sh
# Vérification post-build d'une image Gateway. Lecture seule, aucun
# service redéployé. Usage : tools/verify_image.sh <image:tag> [sha attendu]
set -eu
IMG="${1:?image requise}"; EXPECTED_SHA="${2:-}"
echo "== $IMG"
docker run --rm --network none --entrypoint sh "$IMG" -c '
  set -e
  echo "python : $(python --version 2>&1)"
  . /etc/os-release; echo "os     : $PRETTY_NAME"
  echo "arch   : $(uname -m)"
  echo "sha    : ${AI_GATEWAY_BUILD_SHA:-<VIDE>}"
  pip check && echo "pip check : OK"
  python -c "import app.main" && echo "import app.main : OK"
'
SHA=$(docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMG")
BASE=$(docker inspect --format '{{index .Config.Labels "org.opencontainers.image.base.digest"}}' "$IMG")
echo "label revision    : ${SHA:-<VIDE>}"
echo "label base.digest : $BASE"
echo "label source      : $(docker inspect --format '{{index .Config.Labels "org.opencontainers.image.source"}}' "$IMG")"
if [ -n "$EXPECTED_SHA" ]; then
  [ "$SHA" = "$EXPECTED_SHA" ] || { echo "ERREUR : revision $SHA != attendu $EXPECTED_SHA"; exit 1; }
  echo "revision == sha attendu : OK"
fi
[ -n "$SHA" ] || { echo "ERREUR : image de release sans AI_GATEWAY_BUILD_SHA"; exit 1; }
echo "== versions installées vs lock"
docker run --rm --network none --entrypoint python "$IMG" -m pip freeze | tr 'A-Z_' 'a-z-' | sort > /tmp/verify_freeze.$$
grep -oE '^[A-Za-z0-9_.-]+(\[[a-z,]+\])?==[0-9][^ \\]*' requirements.txt | sed 's/\[[a-z,]*\]//' | tr 'A-Z_' 'a-z-' | sort > /tmp/verify_lock.$$
if diff -u /tmp/verify_lock.$$ /tmp/verify_freeze.$$; then echo "freeze == lock : OK"; else echo "ERREUR : écart freeze/lock"; exit 1; fi
rm -f /tmp/verify_freeze.$$ /tmp/verify_lock.$$
