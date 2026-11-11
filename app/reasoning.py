"""Raisonnement gouverné (lot C1) — abstraction Gateway ↔ `think` Ollama.

Le client ne parle JAMAIS le `think` brut d'Ollama : il demande un mode de
raisonnement abstrait (`reasoning`), la Gateway le confronte au profil du
modèle et ne relaie qu'une valeur validée. Trois familles de modèles
existent réellement (évaluation du 2026-09-26 sur Ollama 0.31.1) :

- `none`   : le modèle n'a pas de mode de raisonnement pilotable (tous les
             modèles du registre embarqué) → aucun champ `think` envoyé ;
- `toggle` : raisonnement activable/désactivable (`think: true/false`),
             sans niveaux — glm-4.7-flash ignore low/medium/high ;
- `levels` : niveaux `low/medium/high` réellement honorés (`think: "low"`…),
             éventuellement NON désactivable — gpt-oss:20b ignore
             `think: false` et réfléchit quand même.

Ce module ne contient que des définitions et des fonctions PURES : aucun
accès réseau, aucun état. La vérité d'un profil vit dans le registre ; ce
qui est relayé à Ollama en découle mécaniquement. Aucun profil concret de
gpt-oss ni de glm n'est déclaré ici ni dans le registre embarqué : C1 pose
le contrat, C2 l'instancie.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

TYPE_NONE = "none"
TYPE_TOGGLE = "toggle"
TYPE_LEVELS = "levels"
PROFILE_TYPES = (TYPE_NONE, TYPE_TOGGLE, TYPE_LEVELS)

# Vocabulaire FERMÉ du contrat client (`reasoning`). Aussi la liste bornée
# des valeurs journalisables : jamais une chaîne libre.
MODE_NONE = "none"
MODE_OFF = "off"
MODE_ON = "on"
MODE_LOW = "low"
MODE_MEDIUM = "medium"
MODE_HIGH = "high"
LEVEL_MODES = (MODE_LOW, MODE_MEDIUM, MODE_HIGH)
DISABLE_MODES = (MODE_NONE, MODE_OFF)
ALL_MODES = (MODE_NONE, MODE_OFF, MODE_ON) + LEVEL_MODES

# Codes d'erreur publics (400), stables.
ERR_NOT_SUPPORTED = "reasoning_not_supported"          # profil none, mode demandé
ERR_MODE_NOT_SUPPORTED = "reasoning_mode_not_supported"  # mode hors profil
ERR_DISABLE_NOT_SUPPORTED = "reasoning_disable_not_supported"  # none/off refusé

# Profil IMPLICITE d'un modèle sans `reasoning_profile` : aucun raisonnement
# pilotable. C'est la situation de tous les modèles du registre embarqué.
IMPLICIT_PROFILE_NAME = "none"


class ReasoningModeError(ValueError):
    """Mode de raisonnement refusé par le profil — AVANT tout appel Ollama."""

    def __init__(self, code: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = dict(extra or {})


@dataclass(frozen=True)
class ReasoningProfile:
    name: str
    type: str
    # Mode Gateway appliqué quand le client ne demande rien. None pour `none`.
    default: Optional[str]
    # Modes acceptés (levels : sous-ensemble ordonné de low/medium/high ;
    # toggle : off/on ; none : none uniquement).
    allowed: Tuple[str, ...]
    # Une demande `none`/`off` est-elle HONORABLE par le modèle ? False pour un
    # modèle qui réfléchit toujours : on refuse en 400 plutôt que d'envoyer
    # `think: false` en espérant qu'il soit respecté.
    disable_supported: bool
    # La trace de réflexion est-elle renvoyée au client ? False par défaut :
    # la Gateway peut exploiter le raisonnement sans en exposer le texte.
    expose_thinking: bool = False

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "default": self.default,
            "allowed": list(self.allowed),
            "disable_supported": self.disable_supported,
            "expose_thinking": self.expose_thinking,
        }


IMPLICIT_NONE = ReasoningProfile(
    name=IMPLICIT_PROFILE_NAME, type=TYPE_NONE, default=None,
    allowed=(MODE_NONE,), disable_supported=True, expose_thinking=False,
)


@dataclass(frozen=True)
class ReasoningDecision:
    """Résultat de `resolve_reasoning` : mode Gateway effectif (borné, ou None
    si aucun raisonnement n'est piloté) et valeur Ollama `think` à relayer
    (None = champ absent)."""

    mode: Optional[str]
    think: Optional[Union[bool, str]]
    requested: bool


class _ProfileError(ValueError):
    pass


def build_reasoning_profile(name: str, raw: Any) -> ReasoningProfile:
    """Construit et VALIDE un profil depuis sa forme YAML. Lève ValueError
    (message nommant le champ) : le registre la convertit en RegistryError."""
    where = f"reasoning_profiles.{name}"
    if not isinstance(raw, dict):
        raise _ProfileError(f"{where} doit être un mapping")
    ptype = raw.get("type")
    if ptype not in PROFILE_TYPES:
        raise _ProfileError(f"{where}.type inconnu : {ptype!r} (admis : {', '.join(PROFILE_TYPES)})")
    expose = raw.get("expose_thinking", False)
    if not isinstance(expose, bool):
        raise _ProfileError(f"{where}.expose_thinking doit être un booléen")
    for key in raw:
        if key not in ("type", "default", "allowed", "disable_supported", "expose_thinking"):
            raise _ProfileError(f"{where}: champ inconnu {key!r}")

    if ptype == TYPE_NONE:
        for key in ("default", "allowed"):
            if raw.get(key) is not None:
                raise _ProfileError(f"{where}.{key} n'a pas de sens pour le type none")
        if raw.get("disable_supported", True) is not True:
            raise _ProfileError(f"{where}.disable_supported ne peut être que true pour le type none")
        return ReasoningProfile(name=name, type=TYPE_NONE, default=None,
                                allowed=(MODE_NONE,), disable_supported=True, expose_thinking=expose)

    disable_supported = raw.get("disable_supported", ptype == TYPE_TOGGLE)
    if not isinstance(disable_supported, bool):
        raise _ProfileError(f"{where}.disable_supported doit être un booléen")

    if ptype == TYPE_TOGGLE:
        if raw.get("allowed") is not None:
            raise _ProfileError(f"{where}.allowed n'a pas de sens pour le type toggle (off/on implicites)")
        default = raw.get("default", MODE_OFF)
        if default not in (MODE_OFF, MODE_ON):
            raise _ProfileError(f"{where}.default doit valoir off ou on (reçu {default!r})")
        if not disable_supported and default != MODE_ON:
            raise _ProfileError(f"{where}: disable_supported=false impose default=on")
        allowed = (MODE_OFF, MODE_ON) if disable_supported else (MODE_ON,)
        return ReasoningProfile(name=name, type=TYPE_TOGGLE, default=default, allowed=allowed,
                                disable_supported=disable_supported, expose_thinking=expose)

    # levels
    allowed_raw = raw.get("allowed")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise _ProfileError(f"{where}.allowed doit être une liste non vide de niveaux")
    if any(level not in LEVEL_MODES for level in allowed_raw) or len(set(allowed_raw)) != len(allowed_raw):
        raise _ProfileError(f"{where}.allowed ne doit contenir que low/medium/high, sans doublon")
    allowed = tuple(level for level in LEVEL_MODES if level in allowed_raw)
    if "default" not in raw:
        raise _ProfileError(f"{where}.default est obligatoire pour le type levels")
    default = raw["default"]
    if default not in allowed:
        raise _ProfileError(f"{where}.default {default!r} n'est pas dans allowed {list(allowed)}")
    return ReasoningProfile(name=name, type=TYPE_LEVELS, default=default, allowed=allowed,
                            disable_supported=disable_supported, expose_thinking=expose)


def map_to_think(profile: ReasoningProfile, mode: Optional[str]) -> Optional[Union[bool, str]]:
    """Mode Gateway (déjà validé) → valeur Ollama `think`. Fonction pure.

        none   : quel que soit le mode → champ absent
        toggle : off → False, on → True
        levels : low/medium/high → "low"/"medium"/"high", none/off → False
    """
    if profile.type == TYPE_NONE or mode is None:
        return None
    if profile.type == TYPE_TOGGLE:
        return mode == MODE_ON
    if mode in DISABLE_MODES:
        return False
    return mode


def resolve_reasoning(profile: ReasoningProfile, requested: Optional[str]) -> ReasoningDecision:
    """Demande client (ou None) → décision validée. Lève ReasoningModeError.

    Vérité de chaque famille (cf. docstring du module) ; aucune valeur n'est
    corrigée en silence : soit elle est honorable, soit c'est un 400.
    """
    if requested is not None and requested not in ALL_MODES:
        raise ReasoningModeError(
            ERR_MODE_NOT_SUPPORTED, "mode de raisonnement inconnu",
            {"reasoning_profile": profile.name, "allowed": list(profile.allowed)},
        )

    if profile.type == TYPE_NONE:
        if requested is None or requested == MODE_NONE:
            return ReasoningDecision(mode=None, think=None, requested=requested is not None)
        raise ReasoningModeError(
            ERR_NOT_SUPPORTED,
            f"le modèle ne pilote aucun mode de raisonnement (profil {profile.name})",
            {"reasoning_profile": profile.name, "requested": requested, "allowed": [MODE_NONE]},
        )

    if requested is None:
        mode = profile.default
        return ReasoningDecision(mode=mode, think=map_to_think(profile, mode), requested=False)

    mode = requested
    if profile.type == TYPE_TOGGLE and mode == MODE_NONE:
        mode = MODE_OFF                      # synonyme : « pas de raisonnement »
    if mode in DISABLE_MODES and not profile.disable_supported:
        raise ReasoningModeError(
            ERR_DISABLE_NOT_SUPPORTED,
            f"le raisonnement de ce modèle n'est pas désactivable (profil {profile.name})",
            {"reasoning_profile": profile.name, "requested": requested, "allowed": list(profile.allowed)},
        )
    if profile.type == TYPE_LEVELS and mode in DISABLE_MODES:
        return ReasoningDecision(mode=MODE_NONE, think=False, requested=True)
    if mode not in profile.allowed:
        raise ReasoningModeError(
            ERR_MODE_NOT_SUPPORTED,
            f"mode de raisonnement {requested!r} non admis par le profil {profile.name}",
            {"reasoning_profile": profile.name, "requested": requested, "allowed": list(profile.allowed)},
        )
    return ReasoningDecision(mode=mode, think=map_to_think(profile, mode), requested=True)
