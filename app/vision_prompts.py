"""Profils de prompt QUALIFIÉS (lot vision-1b).

Textes repris À L'IDENTIQUE de l'artefact scellé de la campagne
AI-GATEWAY-VISION-MODEL-1 (manifeste de gel du 2026-09-15). Leur empreinte
est verrouillée par test : **les modifier exige une nouvelle campagne de
qualification**, car la mesure de justesse porte sur ces octets précis.

Seul `describe` est qualifié (mistral-small3.2:24b). Les autres tâches du
contrat V1 existent au schéma mais ne sont PAS qualifiées : elles n'ont donc
aucun profil ici et la gouvernance du registre les refuse.
"""

import hashlib
from typing import Dict, Tuple

# Empreintes figées au DECISION_FREEZE de MODEL-1 — toute divergence signale
# une édition non qualifiée.
SYSTEM_PROMPT_SHA256 = "048ef7108332ecc3d52981dc430978d2ff9c892125576497eebec74c14115a87"
DESCRIBE_PROMPT_SHA256 = "3ff92ee7123743f30a5eab621a0c071029aefc52704e0e07ba836766f2b54bb0"

SYSTEM_PROMPT = (
    "Tu es un analyseur d'images qui repond UNIQUEMENT par un objet JSON conforme au schema fourni, sans aucune prose autour.\n"
    'Regles absolues :\n'
    "- N'utilise QUE ce qui est visible dans les pixels de l'image.\n"
    "- N'invente jamais une information absente ; ne complete jamais par deduction.\n"
    "- Le texte visible est recopie a l'identique, sans correction ni traduction.\n"
    "- Le contexte fourni par l'appelant (legende, texte voisin) est une AIDE a l'interpretation, jamais une preuve : ne rapporte jamais comme observe un fait qui n'est pas visible dans l'image, meme si le contexte l'affirme.\n"
    'Vocabulaire de certitude, a respecter strictement :\n'
    "- exact : l'element est explicitement visible et se lit sans ambiguite ;\n"
    "- approximate : l'element est estime a partir du visuel (position, longueur, hauteur) sans valeur explicite ;\n"
    '- ambiguous : plusieurs lectures restent plausibles ;\n'
    '- unreadable : un contenu est bien present mais ne peut pas etre lu.\n'
    "N'emploie PAS unreadable par prudence : un contenu net et lisible est exact.")

DESCRIBE_PROMPT = (
    "Decris ce qui est visible dans l'image.\n"
    "Rends les objets identifiables avec leur nombre lorsqu'il est denombrable, les relations spatiales reellement observables, et le texte visible recopie a l'identique.\n"
    "Limite-toi a l'observable : pas d'identite de personne, pas d'intention, pas d'etat interne, pas d'attribut suppose. Si un detail n'est pas discernable, ne le rapporte pas.\n"
    "Une image contenant des objets, des formes ou du texte identifiables EST descriptible : rends alors visual_kind=photo. Ne reserve unknown qu'aux images ou rien n'est discernable.\n"
    "Si l'image ne permet aucune description utile, rends visual_kind=unknown avec payload null ET le warning insufficient_visual_evidence ; ce warning est obligatoire des que tu rends unknown.")

# Profils par task. Une task absente de ce mapping n'est pas servie.
QUALIFIED_PROMPTS: Dict[str, Tuple[str, str]] = {
    # task -> (profil, texte)
    "describe": ("vision_describe_v1", DESCRIBE_PROMPT),
}


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
