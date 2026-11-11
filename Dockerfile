# Image de base ÉPINGLÉE par digest — tag lisible conservé.
# Digest de l'index OCI multi-arch de `python:3.12-slim` (Debian 13 « trixie »),
# relevé sur le registre Docker Hub le 2026-11-09 (`docker buildx imagetools inspect`)
# — lot de qualification : le digest du 2026-09-26 (sha256:423ed6ab…) portait 3 avis
# CRITIQUES corrigés (perl-base) et 57 avis ÉLEVÉS corrigeables (scan trivy) ; le
# digest courant est celui déjà utilisé par l'image backend RAGatm Open (même base).
# Le tag seul est MOBILE : un rebuild sans digest changerait Python à l'insu de tous.
# Renouvellement volontaire : doc/07-developpement.md § « Image de base ».
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Provenance de build (lot build-provenance-observability). Le SHA git est
# injecté PAR LA COMMANDE DE BUILD (cf. doc/05-exploitation.md) :
#   AI_GATEWAY_BUILD_SHA=$(git rev-parse HEAD) docker compose build
# Jamais dérivé d'un `git` exécuté dans l'image : le conteneur n'embarque ni
# .git ni dépôt monté. Défaut vide => l'API rend `not_reported`.
ARG AI_GATEWAY_BUILD_SHA=""
ENV AI_GATEWAY_BUILD_SHA=${AI_GATEWAY_BUILD_SHA}
# URL du dépôt source de CETTE distribution (fournie au build ; vide tant que
# le dépôt Open n'est pas publié — aucune URL privée par défaut).
ARG AI_GATEWAY_SOURCE_URL=""
# Labels OCI de provenance : uniquement des informations RÉELLES et
# déterministes — pas de `created` (un horodatage rendrait la config d'image
# différente à chaque build sans rien prouver de plus ; `Created` reste lu
# dans les métadonnées de l'image). `base.digest` = le digest du FROM.
LABEL org.opencontainers.image.revision=${AI_GATEWAY_BUILD_SHA} \
      org.opencontainers.image.source=${AI_GATEWAY_SOURCE_URL} \
      org.opencontainers.image.base.name="docker.io/library/python:3.12-slim" \
      org.opencontainers.image.base.digest="sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser

# Lock GÉNÉRÉ (pip-compile --generate-hashes depuis requirements.in) :
# versions exactes + hashes de TOUTES les dépendances, directes et
# transitives. `--require-hashes` fait échouer l'installation si un artefact
# ne correspond pas à son hash ou si une dépendance requise n'est pas listée
# rien n'est résolu librement au build.
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

COPY app/ ./app/

USER appuser

EXPOSE 8100

# Sonde Docker STRICTEMENT non inférentielle : /health/live ne
# contacte aucun runtime, ne charge aucun modèle. Les sondes inférentielles
# (/health/ready, /health/embeddings, /health/rerank) sont explicites et ne
# doivent jamais être branchées ici.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health/live', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
