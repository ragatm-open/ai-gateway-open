"""Lot P15 — reproductibilité du build : contrôles hermétiques sur les fichiers.

Ce que ces tests figent :
1. l'image de base Python est épinglée par digest, tag lisible conservé ;
2. l'installation exige les hashes (`--require-hashes`) ;
3. le lock porte versions exactes ET hashes pour chaque dépendance, directe
   comme transitive, sans URL, sans dépôt éditable, sans ligne libre ;
4. le lock est cohérent avec sa source (`requirements.in`) — une version
   changée dans l'un sans régénérer l'autre est détectée ;
5. les labels OCI de provenance sont déclarés.
Aucun réseau, aucun Docker : lecture de fichiers uniquement.
"""

import os
import re

import pytest

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY_BASE_RE = re.compile(r"^FROM python:3\.12-slim@sha256:[0-9a-f]{64}$")
VERSION_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,_-]+\])?==([0-9][^\s\\]*)")
HASH_RE = re.compile(r"^--hash=sha256:[0-9a-f]{64}$")


def _read(name):
    with open(os.path.join(REPO, name), "r", encoding="utf-8") as handle:
        return handle.read()


def _dockerfile_instructions():
    return [l.strip() for l in _read("Dockerfile").splitlines()
            if l.strip() and not l.strip().startswith("#")]


def _parse_lock(name):
    """{nom normalisé: (version, [hashes])} depuis un lock pip-compile."""
    packages, current = {}, None
    for raw in _read(name).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-c ") or line.startswith("-r "):
            continue
        token = line.rstrip("\\").strip()
        m = VERSION_LINE_RE.match(token)
        if m:
            current = m.group(1).lower().replace("_", "-")
            packages[current] = (m.group(3), [])
            continue
        if token.startswith("--hash="):
            assert current is not None, f"hash orphelin dans {name}: {token}"
            assert HASH_RE.match(token), f"hash mal formé dans {name}: {token}"
            packages[current][1].append(token)
            continue
        if token.startswith("--"):
            continue
        pytest.fail(f"ligne non verrouillée dans {name}: {raw!r}")
    return packages


def _parse_direct(name):
    direct = {}
    for raw in _read(name).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-c ") or line.startswith("-r "):
            continue
        m = VERSION_LINE_RE.match(line)
        assert m, f"dépendance directe non épinglée dans {name}: {raw!r}"
        direct[m.group(1).lower().replace("_", "-")] = m.group(3)
    return direct


# --------------------------------------------------------------------------- #
# 1. Image de base
# --------------------------------------------------------------------------- #
def test_python_base_image_is_pinned_by_digest_with_readable_tag():
    froms = [l for l in _dockerfile_instructions() if l.startswith("FROM ")]
    assert len(froms) == 1
    assert PY_BASE_RE.match(froms[0]), froms[0]


def test_python_major_minor_unchanged():
    assert "python:3.12-slim@" in _read("Dockerfile")
    assert "python:3.13" not in _read("Dockerfile")


# --------------------------------------------------------------------------- #
# 2. Installation
# --------------------------------------------------------------------------- #
def test_pip_install_requires_hashes():
    runs = [l for l in _dockerfile_instructions() if l.startswith("RUN") and "pip install" in l]
    assert len(runs) == 1, runs
    assert "--require-hashes" in runs[0]
    assert "-r requirements.txt" in runs[0]
    assert "--no-cache-dir" in runs[0]


# --------------------------------------------------------------------------- #
# 3. Lock complet et hashé
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["requirements.txt", "requirements-dev.txt"])
def test_every_locked_package_has_exact_version_and_hashes(name):
    packages = _parse_lock(name)
    assert packages, name
    for pkg, (version, hashes) in packages.items():
        assert re.fullmatch(r"[0-9][0-9A-Za-z.!+]*", version), (pkg, version)
        assert hashes, f"{pkg}=={version} sans hash dans {name}"


def test_runtime_lock_covers_direct_and_transitive():
    packages = _parse_lock("requirements.txt")
    # Directes (requirements.in) …
    for pkg in ("fastapi", "uvicorn", "requests", "pydantic", "pyyaml", "pillow"):
        assert pkg in packages, pkg
    # … et transitives figées, jamais résolues librement au build.
    for pkg in ("starlette", "pydantic-core", "anyio", "h11", "idna", "urllib3",
                "certifi", "charset-normalizer", "click", "typing-extensions",
                "annotated-types", "httptools", "uvloop", "watchfiles", "websockets",
                "python-dotenv"):
        assert pkg in packages, f"transitive non verrouillée : {pkg}"


def test_lock_matches_its_source_versions():
    """`requirements.in` est la source ; une version changée d'un côté sans
    régénérer l'autre est une dérive."""
    direct = _parse_direct("requirements.in")
    lock = _parse_lock("requirements.txt")
    for pkg, version in direct.items():
        assert pkg in lock, pkg
        assert lock[pkg][0] == version, (pkg, version, lock[pkg][0])
    dev_direct = _parse_direct("requirements-dev.in")
    dev_lock = _parse_lock("requirements-dev.txt")
    for pkg, version in dev_direct.items():
        assert dev_lock[pkg][0] == version, pkg


def test_locks_are_generated_not_handwritten():
    for name in ("requirements.txt", "requirements-dev.txt"):
        head = _read(name).splitlines()[:8]
        assert any("pip-compile" in l for l in head), name
        assert any("--generate-hashes" in l for l in head), name


def test_no_url_or_editable_requirements():
    for name in ("requirements.txt", "requirements-dev.txt", "requirements.in", "requirements-dev.in"):
        for line in _read(name).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith(("-e ", "git+", "http://", "https://", "file:")), (name, line)
            assert "@ " not in stripped, (name, line)


# --------------------------------------------------------------------------- #
# 4. Provenance
# --------------------------------------------------------------------------- #
def test_oci_provenance_labels_declared():
    content = _read("Dockerfile")
    for label in ("org.opencontainers.image.revision=${AI_GATEWAY_BUILD_SHA}",
                  "org.opencontainers.image.source=",
                  "org.opencontainers.image.base.name=",
                  "org.opencontainers.image.base.digest="):
        assert label in content, label
    # Le digest déclaré en label est celui du FROM (valeur citée ou non).
    from_digest = re.search(r"^FROM [^\n]*@sha256:([0-9a-f]{64})", content, re.M).group(1)
    assert re.search(r'org\.opencontainers\.image\.base\.digest="?sha256:' + from_digest + '"?', content)


def test_no_created_timestamp_label():
    """`created` rendrait la config d'image non déterministe : volontairement
    absent (documenté), la provenance passe par revision + base.digest."""
    assert "org.opencontainers.image.created" not in _read("Dockerfile")


# --------------------------------------------------------------------------- #
# 5. Toolchain de génération épinglée
# --------------------------------------------------------------------------- #
# `pip-tools` comme CIBLE d'un `pip install` (après ses options) sans `==`.
# Une mention en prose sur la même ligne qu'un autre `pip install` n'est pas
# une commande.
UNPINNED_PIP_TOOLS = re.compile(r"pip install(?:\s+-{1,2}[\w-]+(?:=\S+)?)*\s+['\"]?pip-tools['\"]?(?!==)(?![\w.-])")
SYFT_EXPECTED = ("v1.52.0", "sha256:500e2d872ac019436926e8322b4fc1f39441d94d21f6f4046c6ff29b30e8cb02")
PIP_TOOLS_EXPECTED = "7.6.1"


def _toolchain():
    values = {}
    for line in _read("tools/toolchain.env").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k] = v.strip()
    return values


def test_toolchain_pins_pip_tools_exactly():
    assert _toolchain()["PIP_TOOLS_VERSION"] == PIP_TOOLS_EXPECTED
    script = _read("tools/update_requirements.sh")
    assert "pip-tools==$PIP_TOOLS_VERSION" in script
    assert not re.search(UNPINNED_PIP_TOOLS, script), "pip-tools non versionné"
    assert ". \"$ROOT/tools/toolchain.env\"" in script


def test_lock_image_is_the_pinned_runtime_image_not_a_bare_tag():
    script = _read("tools/update_requirements.sh")
    # Dérivé du FROM du Dockerfile, jamais un tag mobile codé en dur.
    assert "grep -m1 '^FROM ' Dockerfile" in script
    assert "python:3.12-slim@sha256:*" in script            # garde : refus si non épinglé
    assert not re.search(r"docker run[^\n]*python:3\.12-slim(?!@sha256)", script)


def test_syft_is_pinned_by_tag_and_digest():
    tc = _toolchain()
    m = re.fullmatch(r"anchore/syft:(v[0-9]+\.[0-9]+\.[0-9]+)@(sha256:[0-9a-f]{64})", tc["SYFT_IMAGE"])
    assert m, tc["SYFT_IMAGE"]
    # Le couple tag/digest est celui relevé et vérifié (syft version 1.52.0) :
    # changer le tag sans relever le digest — ou l'inverse — est détecté.
    assert (m.group(1), m.group(2)) == SYFT_EXPECTED
    assert tc["SYFT_VERSION"] == SYFT_EXPECTED[0].lstrip("v")
    sbom = _read("tools/sbom.sh")
    assert '"$SYFT_IMAGE"' in sbom and ". \"$ROOT/tools/toolchain.env\"" in sbom
    assert "latest" not in sbom


def test_no_mobile_tags_in_operational_commands():
    """Aucun `anchore/syft:latest`, aucun `pip install pip-tools` sans version,
    aucun `docker run … python:3.12-slim` sans digest — ni dans les scripts,
    ni dans les commandes de la documentation."""
    files = [os.path.join("tools", f) for f in os.listdir(os.path.join(REPO, "tools"))
             if f.endswith((".sh", ".env"))]
    files += [os.path.join("doc", f) for f in os.listdir(os.path.join(REPO, "doc")) if f.endswith(".md")]
    files += ["README.md"]
    for name in files:
        for n, line in enumerate(_read(name).splitlines(), 1):
            assert "anchore/syft:latest" not in line, (name, n)
            assert not re.search(UNPINNED_PIP_TOOLS, line), (name, n)
            assert not re.search(r"docker run[^\n]*python:3\.12-slim(?!@sha256:)", line), (name, n)
