"""Kennzahlen, Reihen und Verdikte der Demo (reine Funktionen, ohne Streamlit).

Aufbau: `referenz` (LP-Optimum und LP-Plan mit dem eigenen Transportsimplex), `kennzahlen` (Lücke, Totalvariation zum Produktmaß, Einträge, Anteil θ), die Reihen je
Experiment (`reihe_eps`, `reihe_skala`, `reihe_rohe_luecke`, `reihe_produkte`, `reihe_pfad`, `reihe_verzerrung`) und die Verdikte `verdikt_*` als Funktionen
(Art, Text), die die App mit `getattr(st, Art)(Text)` anzeigt. Jede Zahl eines Verdikts stammt aus der Reihe, die es bekommt.

Zählweise: eine Iteration = zwei Matrix-Vektor-Produkte (siehe `snk_sinkhorn`). Skala: ε/OPT mit OPT = LP-Optimum derselben Instanz bei max C = 1."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import snk_constants as C
from snk_eindeutigkeit import ist_eindeutig
from snk_newton import loese_gibbs, loese_gibbs_newton, loese_gibbs_pfad
from snk_scenario import (Instanz, handbeispiel_h1, handbeispiel_h3, make_gitter_instanz, make_instanz, make_pfad)
from snk_simplex import TOL_MENGE, loese_transport, pruefe_loesung
from snk_sinkhorn import loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale, zeilenfehler


# ----------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen und Formatierung
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    quelle: str = C.DEFAULTS["quelle"]
    kosten_art: str = C.DEFAULTS["kosten_art"]
    n: int = C.DEFAULTS["n_slider"]
    eps_rel: float = C.DEFAULTS["eps_rel"]
    seed: int = C.DEFAULTS["seed_input"]
    eps_roh: float = C.DEFAULTS["eps_roh"]
    inst2: int = C.DEFAULTS["inst2"]
    k_iter: int = C.DEFAULTS["k_iter"]
    pfad_art: str = C.DEFAULTS["pfad_art"]
    pfad_T: int = C.DEFAULTS["pfad_T"]
    pfad_t: float = C.DEFAULTS["pfad_t"]
    verz_kosten: str = C.DEFAULTS["verz_kosten"]
    inst5: int = C.DEFAULTS["inst5"]


_ZUSTAND_ZU_FELD = {"quelle": "quelle", "kosten_art": "kosten_art", "n_slider": "n", "eps_rel": "eps_rel", "seed_input": "seed", "eps_roh": "eps_roh", "inst2": "inst2",
                    "k_iter": "k_iter", "pfad_art": "pfad_art", "pfad_T": "pfad_T", "pfad_t": "pfad_t", "verz_kosten": "verz_kosten", "inst5": "inst5"}


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus dem Session State (Schlüssel der Regler); fehlende Schlüssel nehmen den Standardwert."""
    felder = {feld: zustand[key] for key, feld in _ZUSTAND_ZU_FELD.items() if key in zustand}
    for feld in ("n", "seed", "inst2", "k_iter", "pfad_T", "inst5"):
        if feld in felder:
            felder[feld] = int(felder[feld])
    for feld in ("eps_rel", "eps_roh", "pfad_t"):
        if feld in felder:
            felder[feld] = float(felder[feld])
    return Settings(**felder)


def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{stellen}f}".replace(".", ",")


def de_g(x: float, signifikant: int = 3) -> str:
    """Zahl mit Dezimalkomma und höchstens `signifikant` gültigen Stellen (ohne Exponent, wo es lesbar bleibt)."""
    if x == 0:
        return "0"
    text = f"{x:.{signifikant}g}"
    if "e" in text:
        mant, exp = text.split("e")
        return f"{mant.replace('.', ',')}e{int(exp)}"
    return text.replace(".", ",")


def ganz(x: float) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (für Zählungen wie Produkte und Einträge)."""
    return f"{int(round(x)):,}".replace(",", " ")


def band_ganz(b) -> str:
    """Median [Minimum; Maximum] einer Zählung als ganze Zahlen."""
    return f"{ganz(b[0])} [{ganz(b[1])}; {ganz(b[2])}]"


def prozent(x: float, stellen: int = 1) -> str:
    return de(100.0 * x, stellen) + " %"


def eps_text(e: float) -> str:
    """ε/OPT für die Anzeige: höchstens zwei gültige Stellen (1/ln 3 wird 0,91, 2/ln 3 wird 1,8)."""
    return de_g(e, 2)


def band(werte):
    """[Median, Minimum, Maximum] einer Liste (als Liste, damit die Werte unverändert durch JSON gehen)."""
    v = np.asarray(list(werte), float)
    return [float(np.median(v)), float(v.min()), float(v.max())]


def jsonbar(x):
    """Wandelt Ergebnisse in reine Python-Typen um (Listen, Zahlen, Texte): dieselbe Gestalt wie nach dem Laden aus der JSON-Datei."""
    if isinstance(x, dict):
        return {str(k): jsonbar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonbar(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonbar(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    return x


# ----------------------------------------------------------------------------------------------------------------------------------------
# Referenz und Kennzahlen eines Plans
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Referenz:
    opt: float                 # LP-Optimum
    plan_lp: np.ndarray        # eine optimale Ecke
    eintraege_lp: int          # Einträge der Ecke über 1e-12 (höchstens n + m - 1)
    produkt_kosten: float      # Kosten des Produktmaßes a b^T
    loesung: object            # snk_simplex.Loesung (Preise für den Eindeutigkeitstest)


def referenz(inst: Instanz) -> Referenz:
    """LP-Optimum mit dem eigenen Transportsimplex (Regel „steilste“); der unabhängige Prüfer läuft mit und meldet eine Verletzung als Fehler."""
    lo = loese_transport(inst.a, inst.b, inst.kosten, "steilste")
    fehler = pruefe_loesung(inst.a, inst.b, inst.kosten, lo)
    if fehler:
        raise RuntimeError("Transportsimplex: " + "; ".join(fehler))
    return Referenz(lo.kosten, lo.plan, int((lo.plan > TOL_MENGE).sum()), float((np.outer(inst.a, inst.b) * inst.kosten).sum()), lo)


@dataclass(frozen=True)
class Kennzahlen:
    kosten: float
    luecke: float              # (Kosten - OPT) / OPT
    tv: float                  # Totalvariation zum Produktmaß a b^T: halber L1-Abstand
    eintraege: int             # Einträge des Plans über 1e-6
    theta: float | None        # Anteil am Weg vom LP-Optimum zum Produktmaß; None, wenn beide Kosten gleich sind
    abstand_lp: float          # L1-Abstand zur gefundenen LP-Ecke


def kennzahlen(inst: Instanz, ref: Referenz, P: np.ndarray) -> Kennzahlen:
    kosten = float((P * inst.kosten).sum())
    nenner = ref.produkt_kosten - ref.opt
    theta = (kosten - ref.opt) / nenner if nenner > 1e-12 * max(1.0, ref.opt) else None
    return Kennzahlen(kosten, (kosten - ref.opt) / ref.opt if ref.opt > 0 else float("nan"), 0.5 * float(np.abs(P - np.outer(inst.a, inst.b)).sum()),
                      int((P > C.SCHWELLE_EINTRAG).sum()), theta, float(np.abs(P - ref.plan_lp).sum()))


def instanz_der_ansicht(s: Settings) -> Instanz:
    if s.quelle == C.QUELLE_H1:
        return handbeispiel_h1()
    if s.quelle == C.QUELLE_H3:
        return handbeispiel_h3()
    return make_instanz(s.n, s.n, s.kosten_art, s.seed, C.ZWECK_ANSICHT, 0)


@dataclass(frozen=True)
class Ansicht:
    inst: Instanz
    ref: Referenz
    eps: float                 # rohes ε = ε/OPT · OPT
    plan: np.ndarray
    kz: Kennzahlen
    iterationen: int
    produkte: int
    zeilenfehler: float
    konvergiert: bool


def ansicht(s: Settings) -> Ansicht:
    """Die Instanz der Seitenleiste, ihr LP-Plan und der weiche Plan bei ε/OPT (Iteration bis Zeilenfehler 1e-8, höchstens 30 000 Iterationen)."""
    inst = instanz_der_ansicht(s)
    ref = referenz(inst)
    eps = s.eps_rel * ref.opt
    sk = loese_sinkhorn(inst.a, inst.b, inst.kosten, eps, tol=C.TOL, max_iter=C.MAX_ITERATIONEN)
    P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, sk.f, sk.g)
    return Ansicht(inst, ref, eps, P, kennzahlen(inst, ref, P), sk.iterationen, sk.produkte, zeilenfehler(P, inst.a), sk.konvergiert)


@lru_cache(maxsize=64)
def reihen_instanzen(n: int, art: str, anzahl: int, zweck: int = C.ZWECK_REIHE, seed: int = C.SERIEN_SEED):
    """Die festen Instanzen einer Reihe (mit Referenz): Tupel von (Instanz, Referenz). Gleiche Eingabe, gleiche Instanzen."""
    out = []
    for i in range(anzahl):
        inst = make_instanz(n, n, art, seed, zweck, i)
        out.append((inst, referenz(inst)))
    return tuple(out)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: vom scharfen zum weichen Plan
# ----------------------------------------------------------------------------------------------------------------------------------------

def reihe_eps(n: int, art: str, stufen=C.EPS_REIHE, anzahl: int = C.ANZAHL_REIHE):
    """Je ε/OPT-Stufe (Median, Minimum, Maximum über die Instanzen): TV zum Produktmaß, Kosten des Plans in Vielfachen von OPT, Lücke, Einträge über 1e-6,
    Einträge im Verhältnis zur Ecke, Produkte bis Zeilenfehler 1e-8. Gibt eine Liste von Dicts zurück (eine je Stufe)."""
    paare = reihen_instanzen(n, art, anzahl)
    zeilen = []
    for e in stufen:
        tv, fach, luecke, eintr, verh, prod, theta, konv = [], [], [], [], [], [], [], 0
        for inst, ref in paare:
            sk = loese_sinkhorn(inst.a, inst.b, inst.kosten, e * ref.opt, tol=C.TOL, max_iter=C.MAX_ITERATIONEN)
            P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, e * ref.opt, sk.f, sk.g)
            kz = kennzahlen(inst, ref, P)
            tv.append(kz.tv)
            fach.append(kz.kosten / ref.opt)
            luecke.append(kz.luecke)
            eintr.append(kz.eintraege)
            verh.append(kz.eintraege / ref.eintraege_lp)
            prod.append(sk.produkte)
            if kz.theta is not None:
                theta.append(kz.theta)
            konv += int(sk.konvergiert)
        zeilen.append(dict(eps_rel=e, n=n, anzahl=len(paare), konvergiert=konv, tv=band(tv), fach=band(fach), luecke=band(luecke), eintraege=band(eintr),
                           eintraege_verhaeltnis=band(verh), eintraege_lp=band(ref.eintraege_lp for _, ref in paare), produkte=band(prod),
                           theta=band(theta) if theta else None))
    return zeilen


def _zeile_zu(zeilen, e):
    for z in zeilen:
        if abs(z["eps_rel"] - e) < 1e-12:
            return z
    return None


def verdikt_ansicht(an: Ansicht, eps_rel: float):
    """Kernfrage „Wie weich ist der Plan bei diesem ε/OPT?“ für die Instanz der Seitenleiste (Art, Text)."""
    kz = an.kz
    return "info", (f"Bei ε/OPT = {eps_text(eps_rel)} hat der Plan {kz.eintraege} Einträge über 1e-6 (die Ecke des LP {an.ref.eintraege_lp}); die Kostenlücke beträgt {prozent(kz.luecke, 2)}, "
                    f"die Kosten des Plans sind das {de(kz.kosten / an.ref.opt, 2)}-Fache von OPT, der Abstand zum Produktmaß in Totalvariation {de(kz.tv, 3)} "
                    f"({an.inst.n} Lager, {an.inst.m} Kunden, eine Instanz).")


def verdikt_ansicht_produkte(an: Ansicht, eps_rel: float):
    """Kernfrage „Wie viele Produkte kostet er?“ für die Instanz der Seitenleiste (Art, Text)."""
    if not an.konvergiert:
        return "warning", (f"Bei ε/OPT = {eps_text(eps_rel)} ist die Iteration nach {C.MAX_ITERATIONEN:,} Iterationen ({2 * C.MAX_ITERATIONEN:,} Produkten) nicht konvergiert "
                           f"(Zeilenfehler {de_g(an.zeilenfehler, 2)}).".replace(",", "."))
    return "info", (f"Bei ε/OPT = {eps_text(eps_rel)} braucht die Iteration {an.produkte} Matrix-Vektor-Produkte ({an.iterationen} Iterationen) bis zum Zeilenfehler unter 1e-8 "
                    f"({an.inst.n} Lager, {an.inst.m} Kunden, eine Instanz). Wie die Produkte bei kleinerem ε/OPT wachsen, zeigt Experiment 3.")


def verdikt_vom_scharfen_zum_weichen(zeilen, kz_ansicht: Kennzahlen | None = None):
    """Verdikt zu Experiment 1 (Art, Text) aus der Reihe `reihe_eps` mit den Stufen 46 und 0,046."""
    hoch, tief = _zeile_zu(zeilen, 46.0), _zeile_zu(zeilen, 0.046)
    if hoch is None or tief is None:
        return "info", "Die Reihe enthält die Stufen ε/OPT = 46 und 0,046 nicht."
    n, k = hoch["n"], hoch["anzahl"]
    tv = hoch["tv"][0]
    nah = ("liegt der Plan nahe am Produktmaß" if tv < 0.06 else "liegt der Plan im Abstand " + de(tv, 3) + " vom Produktmaß")
    satz1 = (f"Bei ε/OPT = 46 {nah} (Abstand in Totalvariation {de(tv, 3)}, größter Wert {de(hoch['tv'][2], 3)}; n = {n}, Median über {k} Instanzen); "
             f"die Kosten des Plans sind dort das {de(hoch['fach'][0], 1)}-Fache von OPT.")
    verh = tief["eintraege_verhaeltnis"][0]
    satz2 = (f"Bei ε/OPT = 0,046 liegt die Lücke bei {prozent(tief['luecke'][0], 2)}, der Plan hat aber noch das {de(verh, 1)}-Fache der Einträge über 1e-6 der Ecke "
             f"({ganz(tief['eintraege'][0])} gegen {ganz(tief['eintraege_lp'][0])}, Median).")
    return "info", satz1 + " " + satz2


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: die Skala
# ----------------------------------------------------------------------------------------------------------------------------------------

EPS_REL_FEST = 0.07      # festes ε/OPT der Reihe über n


def reihe_skala(eps_roh: float, anzahl: int, art: str, ns=C.N2_STUFEN, seed: int = C.SERIEN_SEED):
    """Je n: LP-Optimum, ε/OPT eines festen rohen ε und die Lücke bei festem ε/OPT = 0,07 (exakter Plan aus dem Newton-Verfahren), je Instanz und als Band."""
    zeilen = []
    for n in ns:
        opts, epsrel, luecken = [], [], []
        for i in range(anzahl):
            inst = make_instanz(n, n, art, seed, C.ZWECK_SKALA, i)
            ref = referenz(inst)
            opts.append(ref.opt)
            epsrel.append(eps_roh / ref.opt)
            erg = loese_gibbs(inst.a, inst.b, inst.kosten, EPS_REL_FEST * ref.opt)
            luecken.append(kennzahlen(inst, ref, erg.plan).luecke if erg.ok else float("nan"))
        gut = [x for x in luecken if np.isfinite(x)]
        zeilen.append(dict(n=n, anzahl=anzahl, opt=band(opts), eps_rel_fest=band(epsrel), luecke=band(gut) if gut else None, nicht_konvergiert=len(luecken) - len(gut),
                           opt_liste=opts, eps_rel_liste=epsrel, luecke_liste=luecken))
    return zeilen


def handbeispiel_luecken():
    """Dasselbe ε/OPT = 1/ln 3 bei H1 und H3 (beide OPT = 1/2, ε = 1/(2 ln 3)): die Lücken (exakt 25 % und (√1701 − 41)/40)."""
    eps = 0.5 * C.EPS_H1
    out = []
    for inst in (handbeispiel_h1(), handbeispiel_h3()):
        ref = referenz(inst)
        erg = loese_gibbs(inst.a, inst.b, inst.kosten, eps)
        out.append(kennzahlen(inst, ref, erg.plan).luecke)
    return tuple(out)


def verdikt_skala(zeilen, eps_roh: float):
    z = {r["n"]: r for r in zeilen}
    kleinst, groesst = z[min(z)], z[max(z)]
    k = kleinst["anzahl"]
    satz1 = (f"Ein festes ε = {de_g(eps_roh, 2)} ist bei n = {kleinst['n']} gleich {de_g(kleinst['eps_rel_fest'][0], 2)} OPT und bei n = {groesst['n']} gleich "
             f"{de_g(groesst['eps_rel_fest'][0], 2)} OPT (Median über {k} Instanzen).")
    mit = [r for r in zeilen if r["luecke"]]
    if mit:
        med = [r["luecke"][0] for r in mit]
        steigt = all(med[i + 1] >= med[i] for i in range(len(med) - 1))
        reihe = ", ".join(f"{prozent(r['luecke'][0], 2)} (n = {r['n']})" for r in mit)
        breit = max(mit, key=lambda r: r["luecke"][2] - r["luecke"][1])
        satz2 = (f"Bei festem ε/OPT = {de_g(EPS_REL_FEST, 2)} liegt die Lücke im Median bei {reihe}; sie "
                 + ("steigt in dieser Messung mit n" if steigt else "ist in dieser Messung nicht monoton in n")
                 + f", und bei n = {breit['n']} reicht sie zwischen den Instanzen von {prozent(breit['luecke'][1], 2)} bis {prozent(breit['luecke'][2], 2)}.")
        if not steigt and [r["n"] for r in zeilen] == list(C.N2_STUFEN):
            satz2 += (f" Ein Anstieg mit n ist bei {k} Instanzen je n nicht sicher zu sehen: in der Kontrollrechnung der README ({C.ANZAHL_KONTROLLE} Instanzen je n, quadrierter Abstand, "
                      f"Zufalls-Seed {C.KONTROLL_SEED}) steigt der Median von {prozent(C.KONTROLL_MEDIAN[0], 2)} (n = {C.N2_STUFEN[0]}) auf {prozent(C.KONTROLL_MEDIAN[-1], 2)} (n = {C.N2_STUFEN[-1]}).")
    else:
        satz2 = ""
    h1, h3 = handbeispiel_luecken()
    satz3 = (f"Dasselbe ε/OPT gibt bei zwei Instanzen verschiedene Lücken: bei den Handbeispielen H1 und H3 mit ε/OPT = {eps_text(C.EPS_H1)} sind es {prozent(h1, 1)} gegen {prozent(h3, 1)}.")
    return "info", " ".join(t for t in (satz1, satz2, satz3) if t)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Produkte und rohe Kostenlücke
# ----------------------------------------------------------------------------------------------------------------------------------------

def rohe_luecke_verlauf(inst: Instanz, ref: Referenz, eps_rel: float, ks):
    """Zeilenfehler und rohe Kostenlücke des Plans nach k Iterationen (Abbruch ohne Konvergenz), für jedes k. Gibt (k, Zeilenfehler, Lücke) als Listen zurück."""
    eps = eps_rel * ref.opt
    fehler, luecken = [], []
    liste = verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, ks)
    for k, f, g in liste:
        P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g)
        fehler.append(zeilenfehler(P, inst.a))
        luecken.append((float((P * inst.kosten).sum()) - ref.opt) / ref.opt)
    return [k for k, _, _ in liste], fehler, luecken


def reihe_rohe_luecke(n: int, art: str, eps_rel: float, ks=C.K_STUFEN, anzahl: int = C.ANZAHL_REIHE):
    """Je k (Median, Minimum, Maximum über die Instanzen): Zeilenfehler und rohe Lücke; dazu die Zahl der Instanzen mit rohen Kosten unter OPT."""
    paare = reihen_instanzen(n, art, anzahl)
    fehler, luecken = [], []
    for inst, ref in paare:
        kk, fe, lu = rohe_luecke_verlauf(inst, ref, eps_rel, ks)
        fehler.append(fe)
        luecken.append(lu)
    fehler, luecken = np.array(fehler), np.array(luecken)
    return dict(n=n, anzahl=len(paare), eps_rel=eps_rel, k=list(kk), zeilenfehler=[band(fehler[:, i]) for i in range(len(kk))],
                luecke=[band(luecken[:, i]) for i in range(len(kk))], negativ=[int((luecken[:, i] < 0).sum()) for i in range(len(kk))])


def reihe_produkte(n: int, art: str, stufen=C.EPS_PRODUKTE, anzahl: int = C.ANZAHL_REIHE):
    """Matrix-Vektor-Produkte bis Zeilenfehler 1e-8 je ε/OPT-Stufe (Median, Minimum, Maximum), nicht konvergierte Instanzen (30 000 Iterationen = 60 000 Produkte),
    und der Faktor zwischen benachbarten Stufen (Verhältnis der Mediane)."""
    paare = reihen_instanzen(n, art, anzahl)
    zeilen = []
    for e in stufen:
        prod, nicht = [], 0
        for inst, ref in paare:
            sk = loese_sinkhorn(inst.a, inst.b, inst.kosten, e * ref.opt, tol=C.TOL, max_iter=C.MAX_ITERATIONEN)
            prod.append(sk.produkte)
            nicht += int(not sk.konvergiert)
        zeilen.append(dict(eps_rel=e, produkte=band(prod), nicht_konvergiert=nicht))
    faktoren = [zeilen[i + 1]["produkte"][0] / zeilen[i]["produkte"][0] for i in range(len(zeilen) - 1)]
    return dict(n=n, anzahl=len(paare), zeilen=zeilen, faktoren=faktoren, eps_faktoren=[stufen[i] / stufen[i + 1] for i in range(len(stufen) - 1)])


def verdikt_produkte(roh, prod, k_zeige: int = 1):
    """Verdikt zu Experiment 3: rohe Lücke nach einer Iteration (immer), nach k_zeige Iterationen (falls verschieden) und Wachstum der Produkte je Stufe."""
    def roh_satz(k):
        i = roh["k"].index(k)
        med = roh["luecke"][i][0]
        richtung = "unter" if med < 0 else "über"
        return (f"liegen die Kosten bei ε/OPT = {eps_text(roh['eps_rel'])} im Median {prozent(abs(med), 1)} {richtung} dem LP-Optimum, in {roh['negativ'][i]} von {roh['anzahl']} Instanzen darunter "
                f"(n = {roh['n']}); die Zeilensummen weichen dann im Median um {de_g(roh['zeilenfehler'][i][0], 2)} (L1) von den Angeboten ab")

    teile = []
    if 1 in roh["k"]:
        teile.append("Nach 1 Iteration (2 Produkte) " + roh_satz(1) + ".")
    if k_zeige != 1 and k_zeige in roh["k"]:
        teile.append(f"Nach {k_zeige} Iterationen ({2 * k_zeige} Produkte) " + roh_satz(k_zeige) + ".")
    f = prod["faktoren"]
    satz = (f"Die Produkte bis Zeilenfehler 1e-8 wachsen bei n = {prod['n']} um den Faktor {de(min(f), 1)} bis {de(max(f), 1)} je Stufe von etwa {de(np.mean(prod['eps_faktoren']), 1)} in ε "
            f"(Median über {prod['anzahl']} Instanzen).")
    nicht = sum(z["nicht_konvergiert"] for z in prod["zeilen"])
    if nicht:
        satz += f" In {nicht} Läufen wurde die Grenze von {C.MAX_ITERATIONEN:,} Iterationen erreicht.".replace(",", ".")
    teile.append(satz)
    return "info", " ".join(teile)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: glatt statt springend
# ----------------------------------------------------------------------------------------------------------------------------------------

SCHWELLE_TRAEGER = 1e-9
PFAD_EPS_REL = (1.0, 0.1, 0.01)      # ε/OPT-Stufen der Tabelle von Experiment 4


def _mittleres_opt(a, b, C0, C1, punkte: int = 41) -> float:
    """Mittleres LP-Optimum entlang des Pfads auf einem festen Raster von `punkte` Werten (unabhängig von der Schrittzahl T)."""
    return float(np.mean([loese_transport(a, b, C0 + t * (C1 - C0), "steilste").kosten for t in np.linspace(0.0, 1.0, punkte)]))


def lp_pfad(a, b, C0, C1, T: int):
    """LP-Pläne entlang C(t) = C0 + t (C1 - C0) an T + 1 Stellen. Gibt (Pläne, Trägerwechsel, größter L1-Sprung, Sprünge je Schritt) zurück.
    Ein Trägerwechsel ist ein Schritt, in dem sich die Menge der Einträge über 1e-9 ändert (nur eine Untergrenze der Basiswechsel)."""
    ts = np.linspace(0.0, 1.0, T + 1)
    plaene = np.array([loese_transport(a, b, C0 + t * (C1 - C0), "steilste").plan for t in ts])
    traeger = plaene > SCHWELLE_TRAEGER
    wechsel = int((traeger[1:] != traeger[:-1]).reshape(T, -1).any(axis=1).sum())
    spruenge = np.abs(plaene[1:] - plaene[:-1]).sum(axis=(1, 2))
    return plaene, wechsel, float(spruenge.max()), spruenge


def weicher_pfad(a, b, C0, C1, T: int, eps: float):
    """Weiche Pläne (exakt, Newton-Verfahren mit Warmstart) entlang des Pfads bei festem rohem ε. Gibt (Pläne, alle konvergiert) zurück."""
    D = C1 - C0
    f = g = None
    erg0 = loese_gibbs(a, b, C0, eps)
    if not erg0.ok:
        return None, False
    f, g = erg0.f, erg0.g
    plaene = [erg0.plan]
    for t in np.linspace(0.0, 1.0, T + 1)[1:]:
        erg = loese_gibbs_newton(a, b, C0 + t * D, eps, f, g)
        if not erg.ok:                                          # Warmstart zu weit weg (großer Schritt bei kleinem ε): neu mit Fortsetzung in ε
            erg = loese_gibbs(a, b, C0 + t * D, eps)
        if not erg.ok:
            return np.array(plaene), False
        f, g = erg.f, erg.g
        plaene.append(erg.plan)
    return np.array(plaene), True


def pfad_ergebnis(a, b, C0, C1, T: int, eps_rels, eps_bezug: float):
    """Alles zu einem Pfad: LP-Seite und je ε/OPT die weiche Seite (Liste von Dicts mit `eps_rel`). `eps_bezug` ist der Wert, mit dem ε/OPT multipliziert wird
    (mittleres OPT entlang des Pfads). ε·L_emp mit L_emp = größter L1-Sprung / größte Änderung der Kosten je Schritt. Die Pläne selbst gibt es nicht zurück (siehe
    `pfad_plaene_bei`); die Sprünge je Schritt stehen auf 6 Stellen gerundet im Ergebnis."""
    _, wechsel, max_lp, sp_lp = lp_pfad(a, b, C0, C1, T)
    dC = float(np.abs(C1 - C0).max()) / T
    weich = []
    for e in eps_rels:
        eps = e * eps_bezug
        plaene, ok = weicher_pfad(a, b, C0, C1, T, eps)
        if plaene is None or not ok:
            weich.append(dict(eps_rel=e, ok=False))
            continue
        sp = np.abs(plaene[1:] - plaene[:-1]).sum(axis=(1, 2))
        weich.append(dict(eps_rel=e, ok=True, eps=eps, max_sprung=float(sp.max()), spruenge=np.round(sp, 6), eps_L=eps * float(sp.max()) / dC))
    return jsonbar(dict(T=T, lp_wechsel=wechsel, lp_max_sprung=max_lp, lp_spruenge=np.round(sp_lp, 6), weich=weich, eps_bezug=eps_bezug))


def pfad_plaene_bei(a, b, C0, C1, t: float, eps: float):
    """Der LP-Plan und der weiche Plan beim Pfadwert t (für die Anzeige). Gibt (LP-Plan, weicher Plan oder None) zurück."""
    Ct = C0 + t * (C1 - C0)
    lp = loese_transport(a, b, Ct, "steilste").plan
    erg = loese_gibbs(a, b, Ct, eps)
    return lp, (erg.plan if erg.ok else None)


def pfad_2x2(T: int, eps_rel: float):
    """Handbeispiel 2 × 2: C(t) = [[t, 1 - t], [1 - t, t]], a = b = (1/2, 1/2). ε = ε/OPT · 1/2 (1/2 ist das Optimum beim Gleichstand t = 1/2)."""
    a = b = np.array([0.5, 0.5])
    C0 = np.array([[0.0, 1.0], [1.0, 0.0]])
    C1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    return pfad_ergebnis(a, b, C0, C1, T, (eps_rel,), 0.5), (a, b, C0, C1)


def pfad_reihe(n: int, art: str, seed: int, T: int, eps_rels=PFAD_EPS_REL, anzahl: int = 5):
    """`anzahl` Pfade 20 × 20 (Zufalls-Seed der Seitenleiste, Zweck Pfad): Ergebnis je Pfad. Der erste Pfad ist der der Anzeige."""
    out = []
    for i in range(anzahl):
        a, b, C0, C1 = make_pfad(n, art, seed, i)
        out.append(pfad_ergebnis(a, b, C0, C1, T, eps_rels, _mittleres_opt(a, b, C0, C1)))
    return out


def verdikt_pfad(ergebnisse, T: int, zwei_mal_zwei: bool = False):
    """Verdikt zu Experiment 4. Für 2 × 2 gibt es einen Pfad (ergebnisse = [Ergebnis]), sonst die Pfade 20 × 20 mit den Stufen PFAD_EPS_REL."""
    if zwei_mal_zwei:
        e = ergebnisse[0]
        w = e["weich"][0]
        if not w.get("ok"):
            return "warning", "Das Newton-Verfahren ist bei diesem ε nicht konvergiert."
        return "info", (f"Der LP-Plan springt im Handbeispiel einmal um {de(e['lp_max_sprung'], 2)} (Höchstwert 2); der weiche Plan hat bei ε = {de_g(w['eps'], 2)} "
                        f"(ε/OPT = {eps_text(w['eps_rel'])}) und T = {T} einen größten Sprung von {de(w['max_sprung'], 3)} und ε·L = {de(w['eps_L'], 2)} "
                        f"(am Gleichstand gilt ε·L = 1 für T gegen unendlich). Glatt gilt nur, solange die Schrittweite klein gegen ε ist.")
    k = len(ergebnisse)
    wechsel = [e["lp_wechsel"] for e in ergebnisse]
    sprung = [e["lp_max_sprung"] for e in ergebnisse]
    teile = []
    for er in PFAD_EPS_REL:
        ws = [w for e in ergebnisse for w in e["weich"] if abs(w["eps_rel"] - er) < 1e-12 and w.get("ok")]
        if ws:
            teile.append((er, band([w["max_sprung"] for w in ws]), max(w["eps_L"] for w in ws)))
    text = (f"LP: mindestens {de(min(wechsel), 0)} bis {de(max(wechsel), 0)} Trägerwechsel entlang eines Pfads, größter Sprung {de(min(sprung), 2)} bis {de(max(sprung), 2)} "
            f"(n = {C.N_PFAD}, T = {T}, {k} Instanzen, Höchstwert 2). Weicher Plan: ")
    text += "; ".join(f"bei ε/OPT = {eps_text(er)} größter Sprung {de(b[1], 3)} bis {de(b[2], 3)} (ε·L höchstens {de(el, 2)})" for er, b, el in teile)
    text += ". Glatt gilt nur, solange die Schrittweite klein gegen ε ist."
    return "info", text


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: wie schnell verschwindet die Verzerrung?
# ----------------------------------------------------------------------------------------------------------------------------------------

LUECKE_BODEN = 1e-11     # darunter ist die Lücke vom Rechenrauschen nicht mehr zu trennen


def verzerrung_instanzen(kosten: str, anzahl: int):
    """Instanzen n = 20 mit s = 0,5 (stetig) oder auf dem 6 × 6-Gitter (Gitterkosten)."""
    if kosten == C.VERZERRUNG_KOSTEN[1]:
        return [make_gitter_instanz(C.N_VERZERRUNG, C.GITTER_SEITE, C.SERIEN_SEED, C.ZWECK_VERZERRUNG, i) for i in range(anzahl)]
    return [make_instanz(C.N_VERZERRUNG, C.N_VERZERRUNG, C.KOSTENARTEN[0], C.SERIEN_SEED, C.ZWECK_VERZERRUNG, i, C.S_PFAD) for i in range(anzahl)]


def reihe_verzerrung(kosten: str, anzahl: int, stufen=C.EPS_VERZERRUNG):
    """Lücke je ε/OPT (exakter Plan, Fortsetzung in ε) für `anzahl` Instanzen. Gibt Median, Spanne, Steigungen der Median-Kurve und die Zahl der Instanzen mit
    eindeutigem Optimum zurück. Nicht konvergierte (Instanz, ε)-Paare fehlen in den Medianen und werden gezählt."""
    insts = verzerrung_instanzen(kosten, anzahl)
    stufen = list(stufen)
    luecke = np.full((len(insts), len(stufen)), np.nan)
    eindeutig = 0
    nicht = 0
    for i, inst in enumerate(insts):
        ref = referenz(inst)
        eindeutig += int(ist_eindeutig(inst.kosten, ref.plan_lp, ref.loesung.u, ref.loesung.v))
        liste = loese_gibbs_pfad(inst.a, inst.b, inst.kosten, [3.0 * ref.opt] + [e * ref.opt for e in stufen])[1:]
        for j, (eps, erg) in enumerate(liste):
            if erg.ok:
                luecke[i, j] = kennzahlen(inst, ref, erg.plan).luecke
            else:
                nicht += 1
    med = np.array([np.nanmedian(luecke[:, j]) if np.isfinite(luecke[:, j]).any() else np.nan for j in range(len(stufen))])
    lo = np.array([np.nanmin(luecke[:, j]) if np.isfinite(luecke[:, j]).any() else np.nan for j in range(len(stufen))])
    hi = np.array([np.nanmax(luecke[:, j]) if np.isfinite(luecke[:, j]).any() else np.nan for j in range(len(stufen))])
    steigung = []
    for j in range(len(stufen) - 1):
        if med[j] > LUECKE_BODEN and med[j + 1] > LUECKE_BODEN:
            steigung.append((math.log(med[j + 1]) - math.log(med[j])) / (math.log(stufen[j + 1]) - math.log(stufen[j])))
        else:
            steigung.append(None)
    halblog = []     # d ln Lücke / d(OPT/ε) zwischen benachbarten Stufen der Median-Kurve: bei exponentiellem Verlauf annähernd konstant
    for j in range(len(stufen) - 1):
        if med[j] > LUECKE_BODEN and med[j + 1] > LUECKE_BODEN:
            halblog.append((math.log(med[j + 1]) - math.log(med[j])) / (1.0 / stufen[j + 1] - 1.0 / stufen[j]))
        else:
            halblog.append(None)
    # je Instanz: ist die Halblog-Anpassung (ln Lücke gegen OPT/ε) besser als die Log-Log-Anpassung? Nur ε/OPT <= 0,1 mit mindestens 4 aufgelösten Punkten
    halblog_besser = bewertet = 0
    x_log = np.log(np.array(stufen))
    x_halb = 1.0 / np.array(stufen)
    for i in range(len(insts)):
        idx = [j for j in range(len(stufen)) if stufen[j] <= 0.1 and np.isfinite(luecke[i, j]) and luecke[i, j] > LUECKE_BODEN]
        if len(idx) >= 4:
            y = np.log(luecke[i, idx])
            r2 = lambda x: float(np.corrcoef(x[idx], y)[0, 1] ** 2)
            bewertet += 1
            halblog_besser += int(r2(x_halb) > r2(x_log))
    return dict(kosten=kosten, anzahl=len(insts), stufen=stufen, median=med.tolist(), minimum=lo.tolist(), maximum=hi.tolist(), steigung=steigung, halblog_steigung=halblog, eindeutig=eindeutig,
                nicht_konvergiert=nicht, halblog_besser=halblog_besser, bewertet=bewertet)


def verdikt_verzerrung(r):
    """Verdikt zu Experiment 5 aus `reihe_verzerrung`."""
    gitter = r["kosten"] == C.VERZERRUNG_KOSTEN[1]
    st = r["stufen"]
    sl = r["steigung"]
    groesse = f"n = {C.N_VERZERRUNG}, {r['anzahl']} Instanzen"
    eind = f"{r['eindeutig']} von {r['anzahl']} Instanzen mit eindeutigem Optimum"
    erste = next((s for s in sl if s is not None), None)
    unter = [s for s, e in zip(sl, st) if s is not None and e <= 0.1]       # Intervalle, die bei ε/OPT <= 0,1 beginnen
    halb = r["halblog_steigung"]
    aufgeloest = [(e, m) for e, m in zip(st, r["median"]) if m > LUECKE_BODEN]
    if erste is None or not unter or not aufgeloest:
        return "info", f"Die Reihe hat zu wenige Punkte über {de_g(LUECKE_BODEN, 1)} ({groesse})."
    halb_werte = [h for h, e in zip(halb, st) if h is not None and e <= 0.32]
    if gitter:
        letzte = aufgeloest[-1]
        satz = (f"Bei Gitterkosten ({C.GITTER_SEITE} × {C.GITTER_SEITE}, {groesse}; {eind}) fällt die Median-Lücke (als Anteil von OPT) von {de_g(r['median'][0], 3)} bei ε/OPT = {eps_text(st[0])} "
                f"auf {de_g(letzte[1], 2)} bei ε/OPT = {eps_text(letzte[0])} (darunter liegt sie unter {de_g(LUECKE_BODEN, 1)}, am Rechenrauschen); "
                f"die Steigung der Median-Kurve wächst von {de(erste, 1)} auf {de(max(s for s in sl if s is not None), 1)}")
        if len(halb_werte) >= 3 and max(abs(h) for h in halb_werte) <= 1.3 * min(abs(h) for h in halb_werte):
            satz += (f", und ln Lücke gegen OPT/ε ist fast linear (Steigung {de(min(halb_werte), 2)} bis {de(max(halb_werte), 2)}): der Verlauf ähnelt einer Exponentialfunktion von 1/ε; ob die Mehrdeutigkeit des Optimums die Ursache ist, ist nicht getestet")
        else:
            satz += ": der Verlauf ist im gemessenen Bereich keine Potenz"
        if r["nicht_konvergiert"]:
            satz += f"; {r['nicht_konvergiert']} nicht konvergierte (Instanz, ε)-Paare fehlen in den Medianen, bei kleinem ε/OPT stützen sie sich auf weniger Instanzen"
        return "info", satz + "."
    satz = (f"Bei eindeutigem Optimum liegt die Steigung der Lückenkurve (d ln Lücke / d ln ε) bei {de(erste, 1)} im ersten Intervall (ε/OPT {eps_text(st[0])} bis {eps_text(st[1])}) "
            f"und bei {de(min(unter), 1)} bis {de(max(unter), 1)} unter ε/OPT = 0,1")
    if max(unter) < 3.0:
        satz += f"; im gemessenen Bereich ε/OPT {eps_text(st[-1])} bis 1 ist der Verlauf der Median-Kurve eine Potenz mit lokaler Steigung {de(min(s for s in sl if s is not None), 1)} bis {de(max(s for s in sl if s is not None), 1)}, ein exponentielles Verhalten (mit 1/ε wachsende Steigung) ist darin nicht zu erkennen"
    else:
        satz += "; die Steigung wächst, ein exponentieller Verlauf ist möglich"
    satz += f" ({groesse}; {eind}"
    if r["bewertet"]:
        satz += f"; Halblog-Anpassung besser als Log-Log in {r['halblog_besser']} von {r['bewertet']} Instanzen"
    return "info", satz + ")."


# ----------------------------------------------------------------------------------------------------------------------------------------
# Vorrechnung für die Standardeinstellungen (siehe snk_vorgerechnet.py)
# ----------------------------------------------------------------------------------------------------------------------------------------

def _vorgerechnet_oder(name: str, params: dict, rechne):
    from snk_vorgerechnet import hole

    treffer = hole(name, **params)
    return treffer if treffer is not None else jsonbar(rechne())


def eps_v(n: int, art: str):
    """Reihe zu Experiment 1 (vorgerechnet für die Standardeinstellung)."""
    return _vorgerechnet_oder("reihe_eps", dict(n=int(n), art=art), lambda: reihe_eps(n, art))


def skala_v(eps_roh: float, anzahl: int, art: str):
    return _vorgerechnet_oder("reihe_skala", dict(eps_roh=float(eps_roh), anzahl=int(anzahl), art=art), lambda: reihe_skala(eps_roh, anzahl, art))


def roh_v(n: int, art: str, eps_rel: float):
    return _vorgerechnet_oder("reihe_roh", dict(n=int(n), art=art, eps_rel=float(eps_rel)), lambda: reihe_rohe_luecke(n, art, eps_rel))


def produkte_v(n: int, art: str):
    return _vorgerechnet_oder("reihe_produkte", dict(n=int(n), art=art), lambda: reihe_produkte(n, art))


def pfad_v(n: int, art: str, seed: int, T: int):
    return _vorgerechnet_oder("pfad_reihe", dict(n=int(n), art=art, seed=int(seed), T=int(T)), lambda: pfad_reihe(n, art, seed, T))


def vorzurechnen():
    """Alle Einträge von tools/vorgerechnet.json als Liste (Name, Parameter, Aufruf). Standard: n = 12, quadrierter Abstand, ε/OPT = 0,46, ε = 0,005, 30 Instanzen,
    Zufalls-Seed 35; die Pfade mit jeder Schrittzahl T (50, 200, 800)."""
    d, art = C.DEFAULTS, C.KOSTENARTEN[0]
    return [
        ("reihe_eps", dict(n=d["n_slider"], art=art), lambda: reihe_eps(d["n_slider"], art)),
        ("reihe_skala", dict(eps_roh=float(d["eps_roh"]), anzahl=d["inst2"], art=art), lambda: reihe_skala(d["eps_roh"], d["inst2"], art)),
        ("reihe_roh", dict(n=d["n_slider"], art=art, eps_rel=float(d["eps_rel"])), lambda: reihe_rohe_luecke(d["n_slider"], art, d["eps_rel"])),
        ("reihe_produkte", dict(n=d["n_slider"], art=art), lambda: reihe_produkte(d["n_slider"], art)),
    ] + [
        ("pfad_reihe", dict(n=C.N_PFAD, art=art, seed=d["seed_input"], T=T), lambda T=T: pfad_reihe(C.N_PFAD, art, d["seed_input"], T)) for T in C.PFAD_T_STUFEN
    ]
