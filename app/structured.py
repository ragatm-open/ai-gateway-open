"""Sorties structurées gouvernées (lot C3) — sous-ensemble JSON Schema,
bornes anti-abus, analyse JSON STRICTE et validation de la réponse.

Le client demande `response_format: {type: "json_schema", schema: {...}}`
sur `/v1/chat`. AI Gateway :

1. refuse tout schéma hors du SOUS-ENSEMBLE ci-dessous ou hors bornes,
   AVANT tout appel Ollama (400) ;
2. relaie le schéma validé tel quel dans le champ `format` de `/api/chat` ;
3. après la réponse, analyse `message.content` STRICTEMENT (aucune
   réparation : pas de fences, pas d'extraction, pas de correction) ;
4. valide l'instance contre le MÊME schéma avec `jsonschema`
   (Draft 2020-12) ; 200 seulement si tout est conforme.

Pourquoi un sous-ensemble : la validation de la réponse s'appuie sur la
bibliothèque `jsonschema`, mais ce qui est ADMIS est décidé ici, par une
liste fermée de mots-clés. Sont refusés explicitement : `$ref` et toute
forme de référence ou de résolution externe (le contrat reste hermétique),
`pattern`/`patternProperties`/`format` (moteur regex sans garantie contre
les expressions pathologiques — différé), les combinateurs
`oneOf`/`anyOf`/`allOf`/`not`/`if` (non activés en C3, support Ollama non
garanti). Aucun mot-clé inconnu n'est ignoré en silence : refus 400.

Fonctions PURES, aucun réseau, aucun état.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

RESPONSE_FORMAT_TYPE = "json_schema"

# ---- bornes anti-abus (constantes, testées, documentées) ------------------- #
SCHEMA_MAX_BYTES = 32 * 1024          # sérialisation compacte UTF-8
SCHEMA_MAX_DEPTH = 16                 # imbrication de schémas (racine = 1)
SCHEMA_MAX_NODES = 512                # nombre de schémas (racine, propriétés, items)
SCHEMA_MAX_PROPERTIES = 256           # total des propriétés déclarées
SCHEMA_MAX_ENUM_VALUES = 256          # total des valeurs d'enum
SCHEMA_MAX_ENUM_STRING_CHARS = 256    # longueur d'une valeur d'enum chaîne
SCHEMA_MAX_REQUIRED = 256             # total des entrées `required`
SCHEMA_MAX_KEY_CHARS = 128            # longueur d'un nom de propriété
SCHEMA_MAX_ANNOTATION_CHARS = 512     # `title` / `description`

SUPPORTED_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
SUPPORTED_KEYWORDS = frozenset({
    "type", "properties", "required", "additionalProperties", "items", "enum", "const",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems",
    "title", "description",
})
# Mots-clés connus et VOLONTAIREMENT refusés (nommés dans l'erreur). Tout
# autre mot-clé est refusé aussi, sans être reproduit dans la réponse.
KNOWN_UNSUPPORTED_KEYWORDS = frozenset({
    "$ref", "$dynamicRef", "$recursiveRef", "$id", "$schema", "$defs", "definitions",
    "$anchor", "$dynamicAnchor", "$comment", "$vocabulary",
    "pattern", "patternProperties", "format", "contentMediaType", "contentEncoding",
    "oneOf", "anyOf", "allOf", "not", "if", "then", "else",
    "dependentSchemas", "dependentRequired", "dependencies", "propertyNames",
    "uniqueItems", "prefixItems", "contains", "minContains", "maxContains",
    "additionalItems", "unevaluatedProperties", "unevaluatedItems",
    "multipleOf", "default", "examples", "deprecated", "readOnly", "writeOnly",
    "minProperties", "maxProperties",
})

# Codes publics (400, provider ai_gateway) — stables.
ERR_INVALID_SCHEMA = "invalid_schema"
ERR_UNSUPPORTED_KEYWORD = "unsupported_schema_keyword"
ERR_SCHEMA_TOO_LARGE = "schema_too_large"

# Codes publics (502, provider ollama) — la réponse du modèle ne satisfait
# pas le contrat demandé.
ERR_OUTPUT_EMPTY = "structured_output_empty"
ERR_OUTPUT_INVALID = "structured_output_invalid"
ERR_OUTPUT_MISMATCH = "structured_output_schema_mismatch"


class SchemaError(ValueError):
    """Schéma client refusé. `message` ne reproduit jamais le schéma."""

    def __init__(self, code: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = dict(extra or {})


class StructuredOutputError(ValueError):
    """Réponse du modèle hors contrat. `message` ne reproduit jamais le
    contenu généré ni le schéma."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SchemaStats:
    """Faits BORNÉS sur le schéma, journalisables : jamais son contenu."""

    bytes: int
    depth: int
    nodes: int
    properties: int
    enum_values: int


def _too_large(what: str, value: int, limit: int) -> SchemaError:
    return SchemaError(
        ERR_SCHEMA_TOO_LARGE,
        f"schéma hors borne : {what} {value} > {limit}",
        {"limit": what, "value": value, "max": limit},
    )


def _invalid(message: str, path: str) -> SchemaError:
    return SchemaError(ERR_INVALID_SCHEMA, f"schéma invalide : {message}", {"path": path})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _is_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    return _is_number(value)


class _Walker:
    def __init__(self) -> None:
        self.nodes = 0
        self.depth = 0
        self.properties = 0
        self.enum_values = 0
        self.required = 0

    def node(self, schema: Any, path: str, depth: int) -> None:
        if not isinstance(schema, dict):
            raise _invalid("un schéma doit être un objet JSON", path)
        self.nodes += 1
        if self.nodes > SCHEMA_MAX_NODES:
            raise _too_large("nodes", self.nodes, SCHEMA_MAX_NODES)
        if depth > SCHEMA_MAX_DEPTH:
            raise _too_large("depth", depth, SCHEMA_MAX_DEPTH)
        self.depth = max(self.depth, depth)

        for keyword in schema:
            if not isinstance(keyword, str):
                raise _invalid("mot-clé non textuel", path)
            if keyword not in SUPPORTED_KEYWORDS:
                shown = keyword if keyword in KNOWN_UNSUPPORTED_KEYWORDS else "unknown"
                raise SchemaError(
                    ERR_UNSUPPORTED_KEYWORD,
                    "mot-clé JSON Schema non supporté dans ce sous-ensemble",
                    {"keyword": shown, "path": path},
                )

        if "type" not in schema:
            raise _invalid("`type` obligatoire sur chaque schéma", path)
        types = self._types(schema["type"], path)

        for key in ("title", "description"):
            if key in schema:
                value = schema[key]
                if not isinstance(value, str) or len(value) > SCHEMA_MAX_ANNOTATION_CHARS:
                    raise _invalid(f"`{key}` doit être une chaîne de {SCHEMA_MAX_ANNOTATION_CHARS} caractères max", path)

        if "enum" in schema:
            self._enum(schema["enum"], path)
        if "const" in schema and not _is_scalar(schema["const"]):
            raise _invalid("`const` doit être un scalaire", path)
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in schema and not _is_number(schema[key]):
                raise _invalid(f"`{key}` doit être un nombre fini", path)
        for key in ("minLength", "maxLength", "minItems", "maxItems"):
            if key in schema and (not _is_int(schema[key]) or schema[key] < 0):
                raise _invalid(f"`{key}` doit être un entier >= 0", path)
        for low, high in (("minLength", "maxLength"), ("minItems", "maxItems"), ("minimum", "maximum")):
            if low in schema and high in schema and schema[low] > schema[high]:
                raise _invalid(f"`{low}` > `{high}`", path)

        if "properties" in schema or "required" in schema or "additionalProperties" in schema:
            if "object" not in types:
                raise _invalid("mots-clés d'objet sur un schéma non `object`", path)
        if "items" in schema and "array" not in types:
            raise _invalid("`items` sur un schéma non `array`", path)

        properties = schema.get("properties", {})
        if "properties" in schema:
            if not isinstance(properties, dict):
                raise _invalid("`properties` doit être un objet", path)
            for name in properties:
                if not isinstance(name, str) or not name or len(name) > SCHEMA_MAX_KEY_CHARS:
                    raise _invalid(f"nom de propriété vide ou > {SCHEMA_MAX_KEY_CHARS} caractères", path)
            self.properties += len(properties)
            if self.properties > SCHEMA_MAX_PROPERTIES:
                raise _too_large("properties", self.properties, SCHEMA_MAX_PROPERTIES)
        if "required" in schema:
            required = schema["required"]
            if not isinstance(required, list) or any(not isinstance(r, str) for r in required):
                raise _invalid("`required` doit être une liste de chaînes", path)
            if len(set(required)) != len(required):
                raise _invalid("`required` contient un doublon", path)
            missing = [r for r in required if r not in properties]
            if missing:
                raise _invalid("`required` référence une propriété non déclarée", path)
            self.required += len(required)
            if self.required > SCHEMA_MAX_REQUIRED:
                raise _too_large("required", self.required, SCHEMA_MAX_REQUIRED)
        if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
            raise _invalid("`additionalProperties` doit être un booléen", path)

        if "properties" in schema:
            for name, sub in properties.items():
                self.node(sub, f"{path}/properties/{name}", depth + 1)
        if "items" in schema:
            self.node(schema["items"], f"{path}/items", depth + 1)

    def _types(self, raw: Any, path: str) -> Tuple[str, ...]:
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, list) and raw and all(isinstance(t, str) for t in raw):
            candidates = list(raw)
        else:
            raise _invalid("`type` doit être un nom de type ou une liste de noms", path)
        if len(set(candidates)) != len(candidates):
            raise _invalid("`type` contient un doublon", path)
        for t in candidates:
            if t not in SUPPORTED_TYPES:
                raise _invalid("type non supporté", path)
        return tuple(candidates)

    def _enum(self, raw: Any, path: str) -> None:
        if not isinstance(raw, list) or not raw:
            raise _invalid("`enum` doit être une liste non vide", path)
        for value in raw:
            if not _is_scalar(value):
                raise _invalid("`enum` ne doit contenir que des scalaires", path)
            if isinstance(value, str) and len(value) > SCHEMA_MAX_ENUM_STRING_CHARS:
                raise _too_large("enum_string_chars", len(value), SCHEMA_MAX_ENUM_STRING_CHARS)
        self.enum_values += len(raw)
        if self.enum_values > SCHEMA_MAX_ENUM_VALUES:
            raise _too_large("enum_values", self.enum_values, SCHEMA_MAX_ENUM_VALUES)


def canonical_schema_bytes(schema: Any) -> bytes:
    """Représentation CANONIQUE d'un schéma : clés triées, séparateurs compacts,
    UTF-8 brut. Base unique de la taille bornée et de l'empreinte
    `response_schema_sha256` (lot G2) — deux schémas équivalents à l'ordre
    des clés près ont la même empreinte."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def serialized_size(schema: Any) -> int:
    return len(canonical_schema_bytes(schema))


def schema_sha256(schema: Any) -> str:
    """Empreinte hexadécimale SHA-256 de la forme canonique du schéma."""
    return hashlib.sha256(canonical_schema_bytes(schema)).hexdigest()


def validate_client_schema(schema: Any) -> SchemaStats:
    """Valide un schéma client contre le sous-ensemble et les bornes.

    Lève `SchemaError` (400) — jamais un appel Ollama avec un schéma refusé.
    Ordre : taille sérialisée d'abord (coût constant), puis parcours borné,
    puis méta-validation `jsonschema` (Draft 2020-12) comme second filet.
    """
    if not isinstance(schema, dict):
        raise _invalid("le schéma doit être un objet JSON", "#")
    try:
        size = serialized_size(schema)
    except (TypeError, ValueError) as exc:
        raise _invalid("schéma non sérialisable en JSON", "#") from exc
    if size > SCHEMA_MAX_BYTES:
        raise _too_large("bytes", size, SCHEMA_MAX_BYTES)
    walker = _Walker()
    walker.node(schema, "#", 1)
    if walker._types(schema["type"], "#") != ("object",):
        raise _invalid("le schéma racine doit être de type `object`", "#")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise _invalid("méta-validation JSON Schema en échec", "#") from exc
    return SchemaStats(
        bytes=size, depth=walker.depth, nodes=walker.nodes,
        properties=walker.properties, enum_values=walker.enum_values,
    )


# ---- réponse du modèle ----------------------------------------------------- #
def _reject_constant(name: str) -> Any:
    # NaN / Infinity / -Infinity ne sont pas du JSON : refus, jamais converti.
    raise ValueError(f"constante non JSON : {name}")


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError("clé dupliquée")
        seen.add(key)
    return dict(pairs)


def parse_strict_json(text: str) -> Any:
    """Analyse STRICTE : exactement un document JSON, entouré au plus de
    blancs. Refuse fences markdown, texte avant/après, documents concaténés,
    document tronqué, NaN/Infinity, clés dupliquées, caractères de contrôle
    bruts. Aucune réparation, aucune extraction."""
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys)


def validate_structured_output(content: str, schema: Dict[str, Any]) -> Any:
    """`message.content` → instance validée, ou StructuredOutputError.

    1. vide/blanc → structured_output_empty ;
    2. JSON strict → structured_output_invalid ;
    3. conformité au schéma (jsonschema, Draft 2020-12, sans format
       checker) → structured_output_schema_mismatch.
    """
    if not isinstance(content, str) or content.strip() == "":
        raise StructuredOutputError(ERR_OUTPUT_EMPTY, "le modèle n'a produit aucun contenu")
    try:
        instance = parse_strict_json(content)
    except ValueError as exc:
        # `str(exc)` d'un JSONDecodeError cite une position, jamais le texte.
        raise StructuredOutputError(ERR_OUTPUT_INVALID, "le contenu n'est pas un document JSON strict") from exc
    validator = jsonschema.Draft202012Validator(schema)
    error = next(validator.iter_errors(instance), None)
    if error is not None:
        raise StructuredOutputError(ERR_OUTPUT_MISMATCH, "le document JSON n'est pas conforme au schéma demandé")
    return instance
