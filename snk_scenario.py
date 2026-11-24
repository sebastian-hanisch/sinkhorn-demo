"""Szenario: Instanzen (Lager mit Angeboten, Kunden mit Nachfragen, Kostenmatrix) aus dem Hausgenerator SplitMix64 (`snk_rng.py`).

Alles ist deterministisch aus (Zufalls-Seed, Zweck, Index): gleiche Eingabe, gleiche Instanz, auf jeder Plattform. Gesamtmasse 1 auf beiden Seiten.
Massen a = (1 - s) / n + s · Dirichlet(1) (Dirichlet(1) = n unabhängige Exponentialzahlen, durch ihre Summe geteilt), b ebenso.
Kosten: quadrierter euklidischer Abstand oder Abstand, jeweils durch den größten Wert geteilt (max C = 1). Gitterkosten: L1-Abstand von Gitterzellen.
Die Handbeispiele H1 und H3 stehen mit ihren Handrechnungen in README und Tests."""

from dataclasses import dataclass

import numpy as np

import snk_constants as C
from snk_rng import MASK, SplitMix64


@dataclass(frozen=True)
class Instanz:
    a: np.ndarray                       # Angebote der n Lager (Summe 1)
    b: np.ndarray                       # Nachfragen der m Kunden (Summe 1)
    kosten: np.ndarray                  # n x m, größter Wert 1
    lager: np.ndarray | None = None     # n x 2 Orte (nur bei geometrischen Instanzen)
    kunden: np.ndarray | None = None    # m x 2 Orte
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def m(self) -> int:
        return len(self.b)


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def massen(g: SplitMix64, n: int, s: float) -> np.ndarray:
    """Angebote bzw. Nachfragen mit Summe 1: (1 - s)/n + s · Dirichlet(1)."""
    if s == 0.0:
        return np.full(n, 1.0 / n)
    e = np.array([g.expovariate(1.0) for _ in range(n)])
    return (1.0 - s) / n + s * e / e.sum()


def kosten_aus_orten(X: np.ndarray, Y: np.ndarray, art: str) -> np.ndarray:
    """Kosten der Orte, durch den größten Wert geteilt. art: quadrierter Abstand oder Abstand."""
    if art not in C.KOSTENARTEN:
        raise ValueError(f"unbekannte Kostenart: {art}")
    q = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    k = q if art == C.KOSTENARTEN[0] else np.sqrt(q)
    return k / k.max()


def make_instanz(n: int, m: int, art: str, seed: int, zweck: int, index: int, s: float = 1.0) -> Instanz:
    """Zufallsinstanz: n Lager und m Kunden gleichverteilt im Einheitsquadrat. Reihenfolge der Ziehungen: Orte der Lager, Orte der Kunden, Massen a, Massen b."""
    if n < 1 or m < 1:
        raise ValueError("mindestens ein Lager und ein Kunde")
    g = instanz_rng(seed, zweck, index)
    X = np.array([[g.random(), g.random()] for _ in range(n)])
    Y = np.array([[g.random(), g.random()] for _ in range(m)])
    a = massen(g, n, s)
    b = massen(g, m, s)
    return Instanz(a, b, kosten_aus_orten(X, Y, art), X, Y, "Zufallsinstanz")


def _zellen_ohne_zuruecklegen(g: SplitMix64, anzahl: int, seite: int) -> np.ndarray:
    """`anzahl` verschiedene Zellen (x, y) eines seite × seite-Gitters (teilweises Mischen)."""
    felder = list(range(seite * seite))
    for k in range(anzahl):
        j = k + g.below(len(felder) - k)
        felder[k], felder[j] = felder[j], felder[k]
    return np.array([(f // seite, f % seite) for f in felder[:anzahl]], dtype=float)


def make_gitter_instanz(n: int, seite: int, seed: int, zweck: int, index: int) -> Instanz:
    """Gitterkosten: n Lager und n Kunden auf je n verschiedenen Zellen eines seite × seite-Gitters, Kosten = L1-Abstand (durch den größten Wert geteilt), Massen 1/n.
    Lager- und Kundenzellen werden unabhängig gezogen (dieselbe Zelle darf bei einem Lager und einem Kunden vorkommen). Es gibt fast immer mehrere optimale Pläne."""
    if n > seite * seite:
        raise ValueError("das Gitter hat nicht genug Zellen")
    g = instanz_rng(seed, zweck, index)
    X = _zellen_ohne_zuruecklegen(g, n, seite)
    Y = _zellen_ohne_zuruecklegen(g, n, seite)
    l1 = np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
    a = np.full(n, 1.0 / n)
    return Instanz(a, a.copy(), l1 / l1.max(), X, Y, "Gitterinstanz")


def handbeispiel_h1() -> Instanz:
    """Handbeispiel H1: zwei Lager, zwei Kunden, Angebote (1/2, 1/2), Nachfragen (1/2, 1/2), Kosten [[1/2, 1], [1, 1/2]]; es hat keine Orte, die Karte zeigt zwei Punktpaare."""
    kosten = np.array([[0.5, 1.0], [1.0, 0.5]])
    lager = np.array([[0.0, 0.0], [0.0, 1.0]])
    kunden = np.array([[1.0, 0.0], [1.0, 1.0]])
    return Instanz(np.array([0.5, 0.5]), np.array([0.5, 0.5]), kosten, lager, kunden, C.QUELLE_H1)


def handbeispiel_h3() -> Instanz:
    """Handbeispiel H3: Angebote (3/4, 1/4), Nachfragen (1/4, 3/4), Kosten [[0, 1], [1, 0]]."""
    kosten = np.array([[0.0, 1.0], [1.0, 0.0]])
    lager = np.array([[0.0, 0.0], [0.0, 1.0]])
    kunden = np.array([[1.0, 0.0], [1.0, 1.0]])
    return Instanz(np.array([0.75, 0.25]), np.array([0.25, 0.75]), kosten, lager, kunden, C.QUELLE_H3)


def make_pfad(n: int, art: str, seed: int, index: int, s: float = C.S_PFAD):
    """Zwei unabhängige Kostenmatrizen C0 und C1 (je Zufallspunkte im Quadrat, max = 1) und gemeinsame Massen: der Pfad C(t) = C0 + t (C1 - C0).
    Gibt (a, b, C0, C1) zurück."""
    g = instanz_rng(seed, C.ZWECK_PFAD, index)
    kosten = []
    for _ in range(2):
        X = np.array([[g.random(), g.random()] for _ in range(n)])
        Y = np.array([[g.random(), g.random()] for _ in range(n)])
        kosten.append(kosten_aus_orten(X, Y, art))
    a = massen(g, n, s)
    b = massen(g, n, s)
    return a, b, kosten[0], kosten[1]
