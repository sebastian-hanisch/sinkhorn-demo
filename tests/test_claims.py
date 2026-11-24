"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Die Messreihen liegen in `tools/messreihen.json` (erzeugt von `tools/messreihen.py` mit dem Code der Demo), die Reihen für die Standardeinstellungen der App in
`tools/vorgerechnet.json`. Dieser Test macht zwei Dinge:
(1) die README-Zahlen werden aus der Datei mit denselben Formatfunktionen wie in der App gebildet und müssen wörtlich in der README stehen (so kann keine Zahl der
    README von der Messung abweichen);
(2) die kleinen Reihen werden neu gerechnet und als Band mit den Dateien verglichen (Zählungen auf 2 % und 2, Gleitkommazahlen auf 1e-6 relativ), weil Gleitkomma-Summen und
    die Wahl der Ecke auf anderen Plattformen abweichen können. Die großen Reihen (n = 40, T = 800) stehen nur in den Dateien."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import snk_constants as C
import snk_evaluation as E
from orakel_zahlen import ORAKEL
from readme_zeilen import J, bt, verzerrung_aufgeloest, zeile_eps, zeile_pfad, zeile_roh, zeile_skala, zeile_verzerrung, zeilen_produkte

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
V = json.loads((ROOT / "tools" / "vorgerechnet.json").read_text(encoding="utf-8"))
QUAD, ABST = C.KOSTENARTEN
ZAEHLFELDER = {"produkte", "eintraege", "eintraege_lp", "negativ", "konvergiert", "nicht_konvergiert", "lp_wechsel", "eindeutig", "halblog_besser", "bewertet", "anzahl"}


def naehe(a, b, schluessel="", rel=1e-6, abs_=1e-12):
    """Band-Vergleich zweier Ergebnisse (verschachtelte Listen und Dicts): Zählfelder auf 2 % und 2, Faktoren aus Zählungen auf 5 %, sonst relativ `rel` (Standard 1e-6);
    Texte und Wahrheitswerte exakt."""
    if isinstance(b, dict):
        assert set(a) == set(b), (schluessel, set(a) ^ set(b))
        for k in b:
            naehe(a[k], b[k], k, rel, abs_)
    elif isinstance(b, list):
        assert len(a) == len(b), schluessel
        for x, y in zip(a, b):
            naehe(x, y, schluessel, rel, abs_)
    elif isinstance(b, bool) or isinstance(b, str) or b is None:
        assert a == b, schluessel
    elif schluessel in ZAEHLFELDER:
        assert abs(a - b) <= max(2.0, 0.02 * abs(b)), (schluessel, a, b)
    elif schluessel in ("faktoren", "eps_faktoren"):
        assert a == pytest.approx(b, rel=0.05), (schluessel, a, b)
    else:
        assert a == pytest.approx(b, rel=rel, abs=abs_), (schluessel, a, b)


def _kurz(x):
    if isinstance(x, dict):
        return {k: _kurz(v) for k, v in x.items() if k not in ("spruenge", "lp_spruenge", "opt_liste", "eps_rel_liste", "luecke_liste")}
    if isinstance(x, list):
        return [_kurz(v) for v in x]
    return x


def par(rows, eps):
    return next(r for r in rows if abs(r["eps_rel"] - eps) < 1e-12)


def minus(text: str) -> str:
    return text.replace("-", "−")


# ----------------------------------------------------------------------------------------------------------------------------------
# Meta und Tabellen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_meta_der_messreihen_ist_der_stand_der_demo():
    assert J["meta"] == {"seed": 2026, "anzahl": 30, "max_iter": 30000, "tol": 1e-8} and C.SERIEN_SEED == 2026 and C.ANZAHL_REIHE == 30
    assert "Zufalls-Seed 2026" in README and "Median [Minimum; Maximum] über 30 Instanzen" in README and "höchstens 30 000 Iterationen" in README and "Zeilenfehler (L1) unter 1e-8" in README
    assert f"({len(C.EPS_REL_STUFEN)} Stufen von 0,01 bis 200)" in README and len(C.EPS_REL_STUFEN) == 29


def test_tabellen_der_readme():
    for key in ("eps_n12", "eps_n40"):
        for r in J[key]:
            assert zeile_eps(r) in README, (key, r["eps_rel"])
    for r in J["skala"]:
        assert zeile_skala(r) in README, r["n"]
    for key in ("roh_n12", "roh_n40"):
        for i, k in enumerate(J[key]["k"]):
            if k in (1, 3, 10, 30, 1000):
                assert zeile_roh(J[key], i) in README, (key, k)
    for key in ("produkte_n12", "produkte_n40"):
        for z in zeilen_produkte(J[key]):
            assert z in README, (key, z)
    for T in C.PFAD_T_STUFEN:
        for er in E.PFAD_EPS_REL:
            assert zeile_pfad(T, er) in README, (T, er)
    for key in ("verzerrung_stetig", "verzerrung_gitter"):
        for j in verzerrung_aufgeloest(J[key]):
            assert zeile_verzerrung(J[key], j) in README, (key, j)


# ----------------------------------------------------------------------------------------------------------------------------------
# Sätze je Experiment
# ----------------------------------------------------------------------------------------------------------------------------------

def test_saetze_experiment_1():
    a, b, ab = J["eps_n12"], J["eps_n40"], J["eps_n12_abstand"]
    assert [r["n"] for r in a] == [12] * 4 and [r["n"] for r in b] == [40] * 4 and all(r["anzahl"] == 30 and r["konvergiert"] == 30 for r in a + b)
    a46, b46, a046, b046 = par(a, 46.0), par(b, 46.0), par(a, 0.046), par(b, 0.046)
    assert (f"Abstand in Totalvariation {E.de_g(a46['tv'][0], 3)} (n = 12) und {E.de_g(b46['tv'][0], 3)} (n = 40) im Median, größter Wert {E.de_g(a46['tv'][2], 3)} und {E.de_g(b46['tv'][2], 3)}") in README
    assert a46["tv"][0] < 0.06 and b46["tv"][0] < 0.06                                                  # „nahe am Produktmaß“: Median der TV unter 0,06
    assert f"das {E.de(a46['fach'][0], 1)}-Fache (n = 12) und das {E.de(b46['fach'][0], 1)}-Fache (n = 40) von OPT" in README
    assert b46["fach"][0] > a46["fach"][0] and "**Das Vielfache wächst mit n**: die Kosten des Plans bei großem ε/OPT sind kein festes Vielfaches von OPT." in README                  # wächst mit n
    assert f"Kostenlücke bei {E.prozent(a046['luecke'][0], 2)} (n = 12) und {E.prozent(b046['luecke'][0], 2)} (n = 40)" in README
    assert (f"**noch das {E.de(a046['eintraege_verhaeltnis'][0], 1)}-Fache (n = 12) und das {E.de(b046['eintraege_verhaeltnis'][0], 1)}-Fache (n = 40) der Einträge über 1e-6 der Ecke** "
            f"({E.ganz(a046['eintraege'][0])} gegen {E.ganz(a046['eintraege_lp'][0])} und {E.ganz(b046['eintraege'][0])} gegen {E.ganz(b046['eintraege_lp'][0])})") in README
    assert a046["eintraege"][0] > a046["eintraege_lp"][0] and b046["eintraege"][0] > b046["eintraege_lp"][0]
    p46, p046 = par(ab, 0.46), par(ab, 0.046)
    assert (f"liegt die Lücke bei ε/OPT = 0,46 und 0,046 bei {E.prozent(p46['luecke'][0], 2)} und {E.prozent(p046['luecke'][0], 2)}, die Einträge bei 0,046 bei "
            f"{E.ganz(p046['eintraege'][0])} gegen {E.ganz(p046['eintraege_lp'][0])}") in README
    assert all(r["tv"][0] < r2["tv"][0] for r, r2 in zip(a, a[1:]))                                     # TV wächst mit kleinerem ε/OPT


def test_saetze_experiment_2():
    z = {r["n"]: r for r in J["skala"]}
    assert sorted(z) == [10, 20, 40, 80] and all(r["anzahl"] == 30 and r["nicht_konvergiert"] == 0 for r in z.values())
    assert f"Ein festes ε = 0,005 ist bei n = 10 gleich {E.de_g(z[10]['eps_rel_fest'][0], 2)} OPT und bei n = 80 gleich {E.de_g(z[80]['eps_rel_fest'][0], 2)} OPT" in README
    med = [z[n]["luecke"][0] for n in (10, 20, 40, 80)]
    assert (f"im Median bei {E.prozent(med[0], 2)} (n = 10), {E.prozent(med[1], 2)} (n = 20), {E.prozent(med[2], 2)} (n = 40) und {E.prozent(med[3], 2)} (n = 80)") in README
    assert not all(med[i + 1] >= med[i] for i in range(3))                                              # „nicht einheitlich“: die Reihe steigt nicht in jedem Schritt
    assert f"(bei n = 20 von {E.prozent(z[20]['luecke'][1], 2)} bis {E.prozent(z[20]['luecke'][2], 1)})" in README
    assert all(z[n]["luecke"][2] - z[n]["luecke"][1] > max(med) - min(med) for n in z)                    # die Streuung innerhalb eines n ist größer als die Medianunterschiede
    h1, h3 = J["hand_luecken"]
    assert f"bei ε/OPT = {E.eps_text(C.EPS_H1)} sind es {E.prozent(h1, 1)} (H1) gegen {E.prozent(h3, 1)} (H3)" in README


def test_saetze_experiment_3():
    r12, r40 = J["roh_n12"], J["roh_n40"]
    assert r12["k"] == r40["k"] == list(C.K_STUFEN) and r12["eps_rel"] == r40["eps_rel"] == 0.46
    i = r12["k"].index(1)
    assert r12["negativ"][i] == r40["negativ"][i] == 30 and r12["anzahl"] == 30
    assert (f"in 30 von 30 Instanzen (n = 12 und n = 40) unter dem LP-Optimum, im Median {E.de(-100 * r12['luecke'][i][0], 1)} % (n = 12) und {E.de(-100 * r40['luecke'][i][0], 1)} % (n = 40); "
            f"die Zeilensummen sind dann um {E.de_g(r12['zeilenfehler'][i][0], 2)} und {E.de_g(r40['zeilenfehler'][i][0], 2)} (L1) falsch") in README
    assert r12["luecke"][i][2] < 0 and r40["luecke"][i][2] < 0                                           # auch die beste Instanz liegt unter OPT
    assert all(r12["negativ"][j] >= r12["negativ"][j + 1] for j in range(len(C.K_STUFEN) - 1))        # „nach mehr Iterationen weniger Instanzen unter OPT“
    p12, p40 = J["produkte_n12"], J["produkte_n40"]
    assert p12["anzahl"] == p40["anzahl"] == 30 and [z["eps_rel"] for z in p12["zeilen"]] == list(C.EPS_PRODUKTE)
    assert sum(z["nicht_konvergiert"] for z in p12["zeilen"] + p40["zeilen"]) == 0
    assert (f"Die Produkte wachsen bei n = 40 um den Faktor {E.de(min(p40['faktoren']), 1)} bis {E.de(max(p40['faktoren']), 1)} je Stufe von 3,0 bis 3,3 in ε, bei n = 12 um "
            f"{E.de(min(p12['faktoren']), 1)} bis {E.de(max(p12['faktoren']), 1)}") in README
    assert min(p40["eps_faktoren"]) > 3.0 and max(p40["eps_faktoren"]) < 3.3 and 3.0 < min(p12["eps_faktoren"])
    assert f"bei n = 12 und ε/OPT = 0,014 reicht sie von {E.ganz(p12['zeilen'][4]['produkte'][1])} bis {E.ganz(p12['zeilen'][4]['produkte'][2])} Produkten" in README
    assert len(p12["zeilen"]) * p12["anzahl"] + len(p40["zeilen"]) * p40["anzahl"] == 300 and "(300 Läufe) erreichte kein Lauf das Limit von 30 000 Iterationen" in README


def test_saetze_experiment_4():
    pf = J["pfad"]
    assert sorted(pf) == ["200", "50", "800"] and all(len(pf[T]) == 5 and all(w["ok"] for e in pf[T] for w in e["weich"]) for T in pf)
    e800 = pf["800"]
    wech, sprung = [e["lp_wechsel"] for e in e800], [e["lp_max_sprung"] for e in e800]
    assert f"bei T = 800 Schritten: {min(wech)} bis {max(wech)}, größter L1-Sprung {E.de(min(sprung), 2)} bis {E.de(max(sprung), 2)} (Höchstwert 2)" in README
    assert "Zufalls-Seed 35" in README and C.DEFAULTS["seed_input"] == 35 and C.N_PFAD == 20

    def med_sprung(T, er):
        return float(sorted(w["max_sprung"] for e in pf[str(T)] for w in e["weich"] if abs(w["eps_rel"] - er) < 1e-12)[2])

    faktoren = [med_sprung(200, er) / med_sprung(800, er) for er in (1.0, 0.1, 0.01)]
    assert f"um den Faktor {E.de(faktoren[0], 1)} bei ε/OPT = 1, {E.de(faktoren[1], 1)} bei 0,1 und {E.de(faktoren[2], 1)} bei 0,01" in README
    el_max = max(w["eps_L"] for T in pf for e in pf[T] for w in e["weich"])
    assert f"ε·L liegt in allen neun Zeilen der Tabelle unter 0,17 (höchstens {E.de(el_max, 3)})" in README and el_max < 0.17
    assert f"mindestens um den Faktor {int(1 / el_max)} unterschritten" in README and int(1 / el_max) == 6
    z = J["pfad_2x2"]
    assert all(z[T]["lp_wechsel"] == 1 and abs(z[T]["lp_max_sprung"] - 2.0) < 1e-12 and z[T]["weich"][0]["eps_rel"] == 0.1 for T in z)
    assert f"ε·L = {E.de(z['50']['weich'][0]['eps_L'], 2)} / {E.de(z['200']['weich'][0]['eps_L'], 2)} / {E.de(z['800']['weich'][0]['eps_L'], 2)} bei T = 50 / 200 / 800" in README
    assert z["800"]["weich"][0]["eps_L"] == pytest.approx(1.0, abs=1e-3) and z["800"]["weich"][0]["eps"] == pytest.approx(0.05)
    lp50 = sorted(e["lp_max_sprung"] for e in pf["50"])[2]
    assert f"der größte Sprung des weichen Plans {E.de(med_sprung(50, 0.01), 3)} (Median), gegen {E.de(lp50, 2)} (Median) beim LP-Plan" in README


def test_saetze_experiment_5():
    s, g = J["verzerrung_stetig"], J["verzerrung_gitter"]
    assert s["anzahl"] == g["anzahl"] == 20 and len(s["stufen"]) == 11 and s["eindeutig"] == 20 and g["eindeutig"] == 0 and s["nicht_konvergiert"] == 0
    assert "20 von 20 Instanzen mit eindeutigem Optimum" in README and "0 von 20 Instanzen mit eindeutigem Optimum" in README
    sl = s["steigung"]
    unter = [x for x, e in zip(sl, s["stufen"]) if x is not None and e <= 0.1]
    assert f"steigt die Steigung der Lückenkurve von {E.de(sl[0], 1)} (ε/OPT von 1 bis 0,56) auf {E.de(min(unter), 1)} bis {E.de(max(unter), 1)} unter ε/OPT = 0,1" in README
    assert max(unter) < 3.0                                                                              # „keine Potenz mit wachsender Steigung“: kein exponentieller Bereich in der Reihe
    lokal = [x for x in sl if x is not None]
    assert f"lokaler Steigung {E.de(min(lokal), 1)} bis {E.de(max(lokal), 1)}" in README and "ein exponentielles Verhalten (eine mit 1/ε wachsende Steigung) ist in diesem Bereich nicht zu erkennen" in README
    assert f"in {s['halblog_besser']} von {s['bewertet']} Instanzen passt die Halblog-Anpassung aber besser" in README
    assert f"{g['nicht_konvergiert']} von {g['anzahl'] * len(g['stufen'])} (Instanz, ε)-Paaren sind nicht konvergiert" in README
    sg = [x for x in g["steigung"] if x is not None]
    j = verzerrung_aufgeloest(g)[-1]
    assert (f"wächst die Steigung von {E.de(sg[0], 1)} auf {E.de(max(sg), 1)} und die Median-Lücke (Anteil von OPT) fällt von {E.de_g(g['median'][0], 3)} auf {E.de_g(g['median'][j], 3)} bei ε/OPT = {E.eps_text(g['stufen'][j])}") in README
    halb = [h for h, e in zip(g["halblog_steigung"], g["stufen"]) if h is not None and e <= 0.32]
    assert max(abs(h) for h in halb) <= 1.3 * min(abs(h) for h in halb)                                   # „fast linear“
    assert f"(Steigung {minus(E.de(min(halb), 2))} bis {minus(E.de(max(halb), 2))})" in README
    assert sg == sorted(sg) and len(set(sg)) == len(sg)                                                   # die Steigung wächst bei Gitterkosten in jeder Stufe
    assert "Ob die Mehrdeutigkeit die Ursache ist, ist nicht getestet" in README


def test_kontrolle_experiment_2_mit_100_instanzen_steigt_mit_n():
    """Die Reihe mit 30 Instanzen steigt nicht monoton in n (Stichprobeneffekt); mit 100 Instanzen je n steigt der Median bei zwei Zufalls-Seeds. Zahlen der README aus der Datei."""
    k26, k777 = J["skala_kontrolle_2026"], J["skala_kontrolle_777"]
    for kontrolle in (k26, k777):
        assert [r["n"] for r in kontrolle] == [10, 20, 40, 80] and all(r["anzahl"] == C.ANZAHL_KONTROLLE == 100 and r["nicht_konvergiert"] == 0 for r in kontrolle)
        med = [r["luecke"][0] for r in kontrolle]
        assert all(med[i] < med[i + 1] for i in range(3))                                               # streng steigend
        assert med[3] - med[0] > 0.004                                                                  # n = 10 gegen n = 80 klar getrennt (Abstand der Mediane, absolut)
    m26, m777 = [r["luecke"][0] for r in k26], [r["luecke"][0] for r in k777]
    assert (f"beim Zufalls-Seed 2026 von {E.prozent(m26[0], 2)} (n = 10) über {E.prozent(m26[1], 2)} (n = 20) und {E.prozent(m26[2], 2)} (n = 40) auf {E.prozent(m26[3], 2)} (n = 80), "
            f"beim Zufalls-Seed {C.KONTROLL_SEED} von {E.prozent(m777[0], 2)} über {E.prozent(m777[1], 2)} und {E.prozent(m777[2], 2)} auf {E.prozent(m777[3], 2)}") in README
    assert C.KONTROLL_SEED == 777 and C.KONTROLL_MEDIAN == pytest.approx(tuple(m777), rel=1e-12)           # die App-Konstante stammt aus der Datei
    z30 = {r["n"]: r["luecke"][0] for r in J["skala"]}
    assert not (z30[10] < z30[20] < z30[40] < z30[80]) and "bei 30 Instanzen sind die Mediane nicht monoton in n" in README


def test_kontrolle_experiment_2_neu_gerechnet_als_band():
    """Zufalls-Seed 777, n = 10 und 80, 100 Instanzen: der Median stimmt mit der Datei auf 1e-6 relativ überein und liegt in der Reihenfolge n = 10 < n = 80."""
    neu = E.jsonbar(E.reihe_skala(0.005, C.ANZAHL_KONTROLLE, QUAD, ns=(10, 80), seed=C.KONTROLL_SEED))
    alt = {r["n"]: r for r in J["skala_kontrolle_777"]}
    for r in neu:
        assert r["luecke"][0] == pytest.approx(alt[r["n"]]["luecke"][0], rel=1e-6) and r["nicht_konvergiert"] == 0
    assert neu[0]["luecke"][0] < neu[1]["luecke"][0]


def test_readme_nennt_tv_und_theta_richtig_und_die_abbruchzaehlung():
    assert "θ fällt mit kleinerem ε und der Abstand zum Produktmaß (TV) wächst mit kleinerem ε" in README and "TV und θ fallen" not in README
    assert "ausgewertet werden also 2k + 1 Log-Sum-Exp bei gemeldeten 2k Produkten" in README and "kostet kein weiteres Produkt" not in README
    assert "Bei ε/OPT = 0,46 liegen die Kosten nach **einer Iteration (zwei Produkte)** in 30 von 30 Instanzen" in README
    assert "Lipschitz-stetig" not in README and "sinkt im gemessenen Bereich mit 1/T" in README


# ----------------------------------------------------------------------------------------------------------------------------------
# Neu gerechnet (Band), Orakel-Zahlen, Testzahl, Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_vorgerechnete_reihen_stimmen_mit_der_neuen_rechnung_ueberein():
    from snk_vorgerechnet import schluessel

    gesehen = set()
    for name, params, rechne in E.vorzurechnen():
        key = schluessel(name, **params)
        gesehen.add(key)
        assert key in V, key
        if name == "pfad_reihe" and params["T"] != 50:
            continue                                                  # T = 200 und 800 dauern 4 und 14 Sekunden: nur in der Datei (Kopf der Datei)
        naehe(E.jsonbar(rechne()), V[key])
    assert gesehen == set(V)


def test_messreihen_n12_und_pfade_gleich_den_vorgerechneten_standardreihen():
    from snk_vorgerechnet import schluessel

    n = {"eps_n12": schluessel("reihe_eps", n=12, art=QUAD), "roh_n12": schluessel("reihe_roh", n=12, art=QUAD, eps_rel=0.46), "produkte_n12": schluessel("reihe_produkte", n=12, art=QUAD),
         "skala": schluessel("reihe_skala", eps_roh=0.005, anzahl=30, art=QUAD)}
    for name, key in n.items():
        naehe(J[name], _kurz(V[key]))
    for T in (50, 200, 800):
        naehe(J["pfad"][str(T)], _kurz(V[schluessel("pfad_reihe", n=20, art=QUAD, seed=35, T=T)]))


def test_messreihe_n40_bei_eps_46_neu_gerechnet_als_band():
    neu = E.reihe_eps(40, QUAD, stufen=(46.0,), anzahl=30)
    naehe(E.jsonbar(neu), [par(J["eps_n40"], 46.0)])


def test_verzerrung_und_pfad_2x2_neu_gerechnet():
    """Stetige Kosten als Band (relativ 1 %, absolut 1e-9). Gitterkosten sind bei kleinem ε schlecht konditioniert (Lücken nahe 1e-12, ein Teil der Paare konvergiert nicht):
    dort nur die Stufen mit ε/OPT >= 0,18 und die Steigungen der ersten Stufen, die Zahl der nicht konvergierten Paare auf ±15."""
    naehe(E.jsonbar(E.reihe_verzerrung(C.VERZERRUNG_KOSTEN[0], 20)), J["verzerrung_stetig"], rel=1e-2, abs_=1e-9)
    neu, alt = E.jsonbar(E.reihe_verzerrung(C.VERZERRUNG_KOSTEN[1], 20)), J["verzerrung_gitter"]
    assert neu["eindeutig"] == alt["eindeutig"] == 0 and neu["stufen"] == alt["stufen"] and abs(neu["nicht_konvergiert"] - alt["nicht_konvergiert"]) <= 15
    for j in range(4):
        assert neu["median"][j] == pytest.approx(alt["median"][j], rel=1e-3) and neu["maximum"][j] == pytest.approx(alt["maximum"][j], rel=1e-3), j
    assert neu["steigung"][:3] == pytest.approx(alt["steigung"][:3], rel=1e-2)
    for T in (50, 200, 800):
        naehe(_kurz(E.jsonbar(E.pfad_2x2(T, 0.1)[0])), J["pfad_2x2"][str(T)])
    naehe(list(E.handbeispiel_luecken()), J["hand_luecken"])


def test_orakel_zahlen():
    d = ORAKEL
    for text in (f"({d['dual']} Zufallsinstanzen bei zwei Werten von ε/OPT, Toleranz 2e-7)", f"(12 Instanzen, 7 Werte von k, relative Toleranz 1e-9)".replace("12", str(d["naiv"])),
                 f"({d['kl']} Instanzen, n = m = 3)", f"(HiGHS, {d['lp']} Instanzen mit n ≠ m)", f"({d['ipm']} Instanzen)", f"({d['netz']} ganzzahlige Instanzen, Abweichung 0)", f"({d['eind']} Instanzen)"):
        assert text in README, text
    assert "mit 200 Zufallsinstanzen" in README


def test_handwerte_der_readme_stehen_in_den_tests():
    for text in ("Kostenlücke 25 %", "θ = ½", "$P = \\begin{pmatrix} 3/8 & 1/8 \\\\ 1/8 & 3/8 \\end{pmatrix}$", "Lücke 1/59 050", "Zeilenfehler 3/10", "rohe Lücke −20 %", "+4,9 %", "0,6 % statt 25 %"):
        assert text in README, text
    test_sinkhorn = (ROOT / "tests" / "test_sinkhorn.py").read_text(encoding="utf-8")
    for beleg in ("Fr(3, 8), Fr(1, 8)", "Fr(1, 59050)", "Fr(-1, 5)", "Fr(3, 10)", "(3 * math.sqrt(3) - 5) / 4", "(math.sqrt(1701) - 41) / 40", "test_h6_sieben_iterationen_sind_vierzehn_produkte"):
        assert beleg in test_sinkhorn, beleg


def test_genannte_tests_der_readme_gibt_es():
    namen = set()
    for pfad in (ROOT / "tests").glob("test_*.py"):
        namen |= set(re.findall(r"^def (test_\w+)", pfad.read_text(encoding="utf-8"), flags=re.M))
    for genannt in set(re.findall(r"`(test_\w+)`", README)):
        assert genannt in namen, genannt


def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"**Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)" in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "trägt", "tragen", "Vormessung", "Mutationstest", "Stück", "V1", "O6", "TODO ", "Plan-"):
            if verboten == "TODO " and text is README:
                continue                                                # der Live-Link-Kommentar bleibt bis zum Deploy
            assert verboten not in text, verboten
    assert "Orakel" not in app and "Zufallsseed" not in README and "Zufalls-Seed" in app
    assert "Sinkhorn-Divergenz" not in app and "Sinkhorn-Distanz" in README and "benutzt das Wort „Sinkhorn-Distanz“ nicht" in README
