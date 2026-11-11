"""Lecture sécurisée d'un jeton depuis un fichier (lot AI-GATEWAY-V2H-SECURITY-ISOLATION-1B-A).

Module autonome et générique (bibliothèque standard seule), présent À L'IDENTIQUE dans la Gateway (`app/`) et dans
le répertoire `backend/` du moteur V2-H : l'image Gateway n'embarque que `app/` et le backend s'exécute en
`python3 -I` depuis sa propre release. Un test vérifie que les deux copies sont octet pour octet identiques.

Contrat :
- le chemin est absolu, normalisé, sans composant `..` ni octet nul ;
- ouverture `O_NOFOLLOW` (un lien symbolique en dernier composant est refusé), puis TOUS les contrôles portent sur
  le descripteur ouvert (`fstat`) : aucune fenêtre entre la vérification et la lecture ;
- fichier régulier, taille <= 4096 octets, aucun droit pour « autres », aucun droit d'écriture pour le groupe,
  propriétaire root ou l'uid effectif du processus (aucun uid numérique codé en dur) ;
- contenu ASCII : un jeton de 32 à 512 caractères `[A-Za-z0-9._~+/=-]`, suivi au plus d'UN saut de ligne `\n`
  final, retiré ; aucune autre transformation (ni `strip`, ni changement de casse).

Toute erreur lève `TokenFileError` dont le message est le seul code : jamais la valeur, ni un extrait, ni une
empreinte, ni le chemin.
"""

import errno
import os
import re
import stat

TOKEN_FILE_NOT_CONFIGURED = "TOKEN_FILE_NOT_CONFIGURED"
TOKEN_FILE_PATH_INVALID = "TOKEN_FILE_PATH_INVALID"
TOKEN_FILE_MISSING = "TOKEN_FILE_MISSING"
TOKEN_FILE_UNREADABLE = "TOKEN_FILE_UNREADABLE"
TOKEN_FILE_NOT_REGULAR = "TOKEN_FILE_NOT_REGULAR"
TOKEN_FILE_SYMLINK_FORBIDDEN = "TOKEN_FILE_SYMLINK_FORBIDDEN"
TOKEN_FILE_PERMISSIONS_INVALID = "TOKEN_FILE_PERMISSIONS_INVALID"
TOKEN_FILE_TOO_LARGE = "TOKEN_FILE_TOO_LARGE"
TOKEN_VALUE_INVALID = "TOKEN_VALUE_INVALID"
LEGACY_ENV_TOKEN_FORBIDDEN = "LEGACY_ENV_TOKEN_FORBIDDEN"
SYSTEMD_CREDENTIAL_DIRECTORY_MISSING = "SYSTEMD_CREDENTIAL_DIRECTORY_MISSING"
SYSTEMD_CREDENTIAL_DIRECTORY_INVALID = "SYSTEMD_CREDENTIAL_DIRECTORY_INVALID"

MAX_FILE_BYTES = 4096
TOKEN_MIN_CHARS = 32
TOKEN_MAX_CHARS = 512
_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/=-]{%d,%d}" % (TOKEN_MIN_CHARS, TOKEN_MAX_CHARS))
# Aucun droit « autres » ; pas d'écriture ni d'exécution pour le groupe ; pas d'exécution pour le propriétaire.
_FORBIDDEN_MODE_BITS = stat.S_IRWXO | stat.S_IWGRP | stat.S_IXGRP | stat.S_IXUSR


class TokenFileError(Exception):
    """Erreur de jeton : `code` est l'unique information exposée (message = code)."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code

    def __repr__(self):
        return f"TokenFileError({self.code!r})"


def _valid_path(path):
    return (isinstance(path, str) and path != "" and "\0" not in path and os.path.isabs(path)
            and os.path.normpath(path) == path and ".." not in path.split("/"))


def read_token_file(path):
    """Lit et valide le jeton contenu dans `path`. Retourne la chaîne du jeton ; lève TokenFileError sinon."""
    if path is None or path == "":
        raise TokenFileError(TOKEN_FILE_NOT_CONFIGURED)
    if not _valid_path(path):
        raise TokenFileError(TOKEN_FILE_PATH_INVALID)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOCTTY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise TokenFileError(TOKEN_FILE_MISSING) from None
        if exc.errno == errno.ELOOP:
            raise TokenFileError(TOKEN_FILE_SYMLINK_FORBIDDEN) from None
        raise TokenFileError(TOKEN_FILE_UNREADABLE) from None
    try:
        try:
            st = os.fstat(fd)
        except OSError:
            raise TokenFileError(TOKEN_FILE_UNREADABLE) from None
        if not stat.S_ISREG(st.st_mode):
            raise TokenFileError(TOKEN_FILE_NOT_REGULAR)
        if st.st_mode & _FORBIDDEN_MODE_BITS or st.st_uid not in (0, os.geteuid()):
            raise TokenFileError(TOKEN_FILE_PERMISSIONS_INVALID)
        if st.st_size > MAX_FILE_BYTES:
            raise TokenFileError(TOKEN_FILE_TOO_LARGE)
        chunks, total = [], 0
        while True:
            try:
                b = os.read(fd, MAX_FILE_BYTES + 1 - total)
            except OSError:
                raise TokenFileError(TOKEN_FILE_UNREADABLE) from None
            if not b:
                break
            chunks.append(b); total += len(b)
            if total > MAX_FILE_BYTES:
                raise TokenFileError(TOKEN_FILE_TOO_LARGE)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise TokenFileError(TOKEN_VALUE_INVALID) from None
    if _TOKEN_RE.fullmatch(text) is None:
        raise TokenFileError(TOKEN_VALUE_INVALID)
    return text


def credential_path(credentials_directory, name):
    """Chemin d'un credential systemd (`LoadCredential=`) : `$CREDENTIALS_DIRECTORY/<name>`, nom FIXE fourni par le
    code. Le répertoire doit être absolu, normalisé, un vrai répertoire (pas un lien), sans écriture pour le groupe
    ni pour « autres », propriétaire root ou l'uid effectif : un environnement altéré ne peut pas le rediriger vers un
    répertoire partagé ou modifiable par un tiers."""
    if credentials_directory is None or credentials_directory == "":
        raise TokenFileError(SYSTEMD_CREDENTIAL_DIRECTORY_MISSING)
    if not _valid_path(credentials_directory) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name or ""):
        raise TokenFileError(SYSTEMD_CREDENTIAL_DIRECTORY_INVALID)
    try:
        st = os.lstat(credentials_directory)
    except OSError:
        raise TokenFileError(SYSTEMD_CREDENTIAL_DIRECTORY_MISSING) from None
    if (not stat.S_ISDIR(st.st_mode) or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or st.st_uid not in (0, os.geteuid())):
        raise TokenFileError(SYSTEMD_CREDENTIAL_DIRECTORY_INVALID)
    return os.path.join(credentials_directory, name)
