"""Provenance technique du rail Vision (lot CV-1) — identité du pipeline.

Répond, des mois plus tard, à « AVEC QUOI cette observation a-t-elle été
produite ? » : un payload canonique qui ne porte QUE ce qui façonne la sortie
du modèle (task, digest physique, octets des prompts, schéma imposé au
modèle, normaliseur, options d'inférence et politique de retry), haché en
`vision-v1:<64 hex>`. Même doctrine que `emb-v2` (`registry.canonical_json`) :
un renommage logique ne change pas l'identité ; un octet de prompt la change.

EXCLUS, par décision (lot CV-1 §27) : nom logique et `ollama_name`
(conséquences du digest), `locale` (réservé, sans effet en V1), `keep_alive`,
timeout, concurrence, environnement, host/port, `schema_version` publique,
`request_id`, contenu du contexte, image, version de Pillow (prouvée par les
vecteurs de test, jamais déclarée). Production et staging ont donc le MÊME
fingerprint.

Toutes les valeurs viennent du code et du registre : aucune lecture de
`calibration/` au runtime (l'archive MODEL-1 est une preuve, pas une config).
Aucune I/O ici.
"""

import hashlib
import json
from typing import Any, Dict

from .registry import canonical_json
from .vision_backend import (
    CONTEXT_WRAPPER_VERSION,
    QUALIFIED_FIRST_ATTEMPT,
    QUALIFIED_RETRY_ATTEMPT,
    RETRY_MAX_ATTEMPTS,
    RETRY_TRIGGER,
    VisionInferenceOptions,
)
from .vision_image import NORMALIZER_ID
from .vision_prompts import SYSTEM_PROMPT, sha256_of
from .vision_schemas import output_schema_for_task

FINGERPRINT_SCHEMA = 1
FINGERPRINT_PREFIX = "vision-v1"
# Convention HISTORIQUE de MODEL-1 (decision_freeze.json.schema_sha256) :
# `json.dumps(schema, sort_keys=True)` avec les séparateurs PAR DÉFAUT — pas
# le JSON compact de canonical_json. Nommée pour ne jamais être réinterprétée.
MODEL_SCHEMA_HASH_ALGORITHM = "json-sortkeys/1"


def model_schema_sha256(task: str) -> str:
    """SHA-256 du JSON Schema imposé au modèle pour `task` (convention MODEL-1)."""
    schema_json = json.dumps(output_schema_for_task(task), sort_keys=True)
    return hashlib.sha256(schema_json.encode("utf-8")).hexdigest()


def pipeline_identity(
    task: str,
    model_digest: str,
    task_prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    first_attempt: VisionInferenceOptions = QUALIFIED_FIRST_ATTEMPT,
    retry_attempt: VisionInferenceOptions = QUALIFIED_RETRY_ATTEMPT,
    retry_max_attempts: int = RETRY_MAX_ATTEMPTS,
    retry_trigger: str = RETRY_TRIGGER,
    normalizer_id: str = NORMALIZER_ID,
    context_wrapper: str = CONTEXT_WRAPPER_VERSION,
    schema_sha256: str = None,
) -> Dict[str, Any]:
    """Payload canonique de l'identité SÉMANTIQUE (lot CV-1 §18). Les
    paramètres nommés ne servent qu'aux tests de mutation : le runtime
    appelle avec les constantes qualifiées."""
    return {
        "schema": FINGERPRINT_SCHEMA,
        "task": task,
        "model_digest": model_digest,
        "system_prompt_sha256": sha256_of(system_prompt),
        "task_prompt_sha256": sha256_of(task_prompt),
        "context_wrapper": context_wrapper,
        "model_schema_sha256": schema_sha256 if schema_sha256 is not None else model_schema_sha256(task),
        "model_schema_hash_algorithm": MODEL_SCHEMA_HASH_ALGORITHM,
        "normalizer_id": normalizer_id,
        "inference": {
            "temperature": first_attempt.temperature,
            "seed": first_attempt.seed,
            "num_ctx": first_attempt.num_ctx,
            "num_predict": first_attempt.num_predict,
        },
        "retry": {
            "max_attempts": retry_max_attempts,
            "trigger": retry_trigger,
            "temperature": retry_attempt.temperature,
            "seed": retry_attempt.seed,
        },
    }


def vision_pipeline_fingerprint(identity: Dict[str, Any]) -> str:
    """`vision-v1:<64 hex>` = SHA-256 complet du payload canonique (jamais tronqué)."""
    material = canonical_json(identity).encode("utf-8")
    return f"{FINGERPRINT_PREFIX}:{hashlib.sha256(material).hexdigest()}"


def pipeline_provenance(task: str, model_digest: str, task_prompt: str) -> Dict[str, Any]:
    """Champs PUBLICS V2 de `pipeline` (hors `prompt_profile*` et `pipeline_id`,
    conservés par la route). Le digest complet n'y est pas dupliqué : il
    reste dans `model.digest` ; il entre dans le fingerprint."""
    identity = pipeline_identity(task, model_digest, task_prompt)
    return {
        "fingerprint": vision_pipeline_fingerprint(identity),
        "system_prompt_sha256": identity["system_prompt_sha256"],
        "task_prompt_sha256": identity["task_prompt_sha256"],
        "model_schema_sha256": identity["model_schema_sha256"],
        "model_schema_hash_algorithm": MODEL_SCHEMA_HASH_ALGORITHM,
        "inference": dict(identity["inference"]),
        "retry": dict(identity["retry"]),
    }
