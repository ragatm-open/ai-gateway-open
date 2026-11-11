"""Admission du rail Vision (lot CV-CAPACITY) : UN objet possède le sémaphore,
le compteur de requêtes en vol et l'instantané de capacité.

Avant ce lot, la route tenait un `threading.BoundedSemaphore` opaque et une
jauge séparée (`gauges["vision:<logical>"]`), incrémentée quelques lignes
après l'acquisition : aucune lecture cohérente de la capacité n'était
possible sans introspecter `_value`, attribut privé hors contrat Python.

Ici `try_acquire()` et `release()` mettent à jour le compteur SOUS LE MÊME
VERROU que celui qui garde l'instantané : `snapshot()` ne peut jamais
observer un slot pris sans compteur, ni l'inverse.

Portée : cet objet vit en mémoire du processus. Le runtime est UN processus
uvicorn par instance (prod :8100, staging :8200), donc « processus » ==
« instance ». Si des workers uvicorn multiples apparaissaient, chaque worker
aurait sa propre admission et la sonde `/v1/vision/capacity` ne verrait que
le sien : le contrat (`scope: "instance"`) devra être revu AVANT tout
scale-out. Prod et staging partagent déjà le GPU et Ollama : la capacité
mesurée ici est celle de CETTE instance, jamais celle de l'hôte.

La sonde est ADVISORY : lire `snapshot()` ne réserve rien ; seul
`try_acquire()` au moment du POST fait autorité.
"""

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger("ai_gateway.vision_admission")


class VisionAdmissionError(RuntimeError):
    """Violation d'invariant (release sans acquisition) — jamais clampée en
    silence : c'est un bug d'appariement, pas une réalité d'exploitation."""


@dataclass(frozen=True)
class VisionAdmissionSnapshot:
    """Photographie atomique : `0 <= in_flight <= max_concurrency`,
    `available_slots == max(0, max_concurrency - in_flight)`."""

    max_concurrency: int
    in_flight: int
    available_slots: int


class VisionAdmission:
    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._in_flight = 0

    def try_acquire(self) -> bool:
        """Acquisition NON bloquante : un slot libre est pris et compté en vol
        dans la même section critique ; sinon False (la route rend 503)."""
        with self._lock:
            if not self._semaphore.acquire(blocking=False):
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        """Libère exactement un slot acquis. Un release sans acquisition est
        une violation d'invariant : journalisée et levée, jamais absorbée."""
        with self._lock:
            if self._in_flight <= 0:
                logger.error("vision_admission: release sans acquisition (in_flight=0)")
                raise VisionAdmissionError("release sans acquisition")
            self._in_flight -= 1
            self._semaphore.release()

    def snapshot(self) -> VisionAdmissionSnapshot:
        """Lecture seule, sans effet de bord, cohérente sous le verrou."""
        with self._lock:
            in_flight = self._in_flight
            return VisionAdmissionSnapshot(
                max_concurrency=self.max_concurrency,
                in_flight=in_flight,
                available_slots=max(0, self.max_concurrency - in_flight),
            )
