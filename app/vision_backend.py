"""Seam d'inférence vision (lot vision-1a).

`VisionBackend` est l'interface que le lot 1B remplacera par le transport
multimodal Ollama réel, SANS toucher à l'API publique, aux schémas, à la
validation ni aux erreurs. En 1A, l'implémentation par défaut refuse de
servir (aucun chemin live accidentel) ; les tests injectent un mock.

Le plan reçu est déjà NORMALISÉ (aucune provenance source : ni PDF, ni page,
ni fichier, ni RagSpace) et le backend rend une sortie STRUCTURÉE BRUTE
(texte JSON) : le parse/validation appartient à la route (étages 2-4).
"""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from .errors import UpstreamError


class VisionBackendTimeout(Exception):
    """Budget mural dépassé pendant l'inférence."""


@dataclass(frozen=True)
class VisionInferenceOptions:
    """Options d'inférence du PROFIL QUALIFIÉ (lot vision-1b).

    Elles font partie du comportement mesuré par MODEL-1 : les changer exige
    une requalification, elles ne sont donc ni exposées au client ni
    surchargeables par l'exploitation (contrat §103).
    """

    temperature: float
    seed: int
    num_ctx: int = 8192
    num_predict: int = 2048
    keep_alive: int = 0        # le modèle vision ne co-réside pas avec le chat


# Profil d'inférence qualifié. Première tentative DÉTERMINISTE (décodage
# glouton, reproductible) ; le retry structurel doit varier la TEMPÉRATURE :
# MODEL-1 a mesuré qu'à temperature=0 le seed n'a aucun effet, un retry
# seed-only reproduisant alors exactement la même sortie invalide.
QUALIFIED_FIRST_ATTEMPT = VisionInferenceOptions(temperature=0.0, seed=42)
QUALIFIED_RETRY_ATTEMPT = VisionInferenceOptions(temperature=0.35, seed=43)

# Politique de retry QUALIFIÉE, nommée pour l'identité du pipeline (lot CV-1) :
# au plus deux appels backend, le second UNIQUEMENT sur sortie structurellement
# invalide (parse/schéma/invariants) ou tronquée — jamais sur échec transport.
# La route (`main.vision_analyze`) implémente exactement cette politique ; ces
# constantes la décrivent, elles ne la pilotent pas.
RETRY_MAX_ATTEMPTS = 2
RETRY_TRIGGER = "structural_invalid_or_truncated"

# Version VOLONTAIRE du texte fixe qui enveloppe le contexte de l'appelant
# (`OllamaVisionBackend._context_block`). Le texte lui-même façonne le prompt
# envoyé : le modifier exige un bump de cette version (le contenu variable du
# contexte — caption, alt_text, nearby_text — est un INPUT, hors identité).
CONTEXT_WRAPPER_VERSION = "1"


@dataclass(frozen=True)
class VisionInferencePlan:
    """Entrée du backend — champs gelés par le contrat (§110)."""

    image_bytes: bytes          # PNG normalisé, sans métadonnées
    image_mime: str             # "image/png"
    task: str
    prompt_profile: str
    prompt_profile_version: str
    output_json_schema: dict    # généré des modèles Pydantic (contrainte modèle)
    locale: str
    caption: Optional[str]
    alt_text: Optional[str]
    nearby_text: Optional[str]
    model_logical: str
    model_concrete: str
    remaining_timeout_seconds: float
    # Ajouts lot vision-1b : ce que le transport réel doit envoyer.
    system_prompt: str = ""
    task_prompt: str = ""
    options: VisionInferenceOptions = QUALIFIED_FIRST_ATTEMPT


@dataclass(frozen=True)
class VisionBackendResult:
    raw_json: str                       # sortie structurée BRUTE (à parser)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    truncated: bool = False             # plafond de sortie atteint côté backend


class VisionBackend:
    """Interface. 1B : transport Ollama multimodal réel (images[] + format)."""

    def analyze(self, plan: VisionInferencePlan) -> VisionBackendResult:
        raise NotImplementedError


class OllamaVisionBackend(VisionBackend):
    """Transport multimodal réel vers Ollama (lot vision-1b).

    PUR TRANSPORT, même doctrine que `ollama_client` : il n'interprète pas la
    sortie, ne la valide pas, ne réessaie pas. Le parse, la validation de
    schéma, les invariants et l'unique retry structurel appartiennent à la
    route — qui seule connaît le contrat V1.

    Le contexte de l'appelant est joint au prompt de tâche dans une section
    explicitement NON PROBANTE : il ne rejoint jamais les instructions
    système, qui portent les règles de confiance.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        # Point d'injection d'instance : les tests remplacent CE transport,
        # sans toucher au `requests` global partagé par les autres clients.
        self._post = requests.post

    @staticmethod
    def _context_block(plan: VisionInferencePlan) -> str:
        lines = [(label, value) for label, value in
                 (("caption", plan.caption), ("alt_text", plan.alt_text),
                  ("nearby_text", plan.nearby_text)) if value]
        if not lines:
            return ""
        body = "\n".join(f"{label} : {value}" for label, value in lines)
        return ("\n\nContexte fourni par l'appelant (AIDE a l'interpretation, "
                "JAMAIS une preuve visuelle) :\n" + body)

    def build_payload(self, plan: VisionInferencePlan) -> Dict[str, Any]:
        """Payload /api/generate exact. Extrait pour être testable sans réseau."""
        return {
            "model": plan.model_concrete,
            "prompt": plan.task_prompt + self._context_block(plan),
            "system": plan.system_prompt,
            "images": [base64.b64encode(plan.image_bytes).decode("ascii")],
            "stream": False,
            "format": plan.output_json_schema,
            "keep_alive": plan.options.keep_alive,
            "options": {
                "temperature": plan.options.temperature,
                "seed": plan.options.seed,
                "num_ctx": plan.options.num_ctx,
                "num_predict": plan.options.num_predict,
            },
        }

    def analyze(self, plan: VisionInferencePlan) -> VisionBackendResult:
        timeout = max(0.1, float(plan.remaining_timeout_seconds))
        try:
            resp = self._post(f"{self.base_url}/api/generate",
                              json=self.build_payload(plan), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout as exc:
            # Budget mural épuisé : distinct d'une panne de transport.
            raise VisionBackendTimeout("budget vision épuisé côté upstream") from exc
        except requests.HTTPError as exc:
            # Le corps d'erreur n'est JAMAIS relayé : il peut citer le prompt.
            status = exc.response.status_code if exc.response is not None else "?"
            raise UpstreamError(f"échec /api/generate vision: HTTP {status}") from exc
        except requests.RequestException as exc:
            raise UpstreamError(f"échec /api/generate vision: {type(exc).__name__}") from exc
        except ValueError as exc:      # corps non JSON
            raise UpstreamError("réponse /api/generate vision illisible") from exc
        if not isinstance(data, dict):
            raise UpstreamError("réponse /api/generate vision inattendue")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total = (prompt_tokens + completion_tokens
                 if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                 else None)
        return VisionBackendResult(
            raw_json=data.get("response") or "",
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total,
            # Plafond de sortie atteint : sortie tronquée, jamais un succès.
            truncated=data.get("done_reason") == "length",
        )


class UnimplementedVisionBackend(VisionBackend):
    """Conservé pour les tests : refuse tout chemin live."""

    def analyze(self, plan: VisionInferencePlan) -> VisionBackendResult:
        raise UpstreamError("transport multimodal non disponible")


from .settings import settings  # noqa: E402  (évite un cycle à l'import)

vision_backend: VisionBackend = OllamaVisionBackend(settings.ollama_base_url)
