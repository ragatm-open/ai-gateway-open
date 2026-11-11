#!/usr/bin/env python3
"""Diagnostic : affiche, pour chaque profil d'embedding du registre,
le payload canonique et le `pipeline_id` qui en dérive — sans aucun appel
réseau ni donnée utilisateur. À utiliser avant de modifier un préfixe, une
politique de troncature ou PREPROCESSING_VERSION : si un identifiant change,
la collection concernée doit être réindexée.

Usage :  python3 tools/pipeline_identity.py [chemin/registre.yaml]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.registry import canonical_json, load_registry  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "app/models_registry.yaml"
for logical, profile in load_registry(path).models.items():
    for task in ("query", "document"):
        print(f"{logical:18} {task:9} {profile.pipeline_id(task)}")
        print(f"{'':28} {canonical_json(profile.pipeline_identity(task))}")
