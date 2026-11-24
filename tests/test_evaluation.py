"""Kennzahlen, Referenz, Reihen und Verdikte (reine Funktionen): Handwerte H1 und H3, Verdikte an handgesetzten Zeilen (Schwellen, jede Zahl im Text, Zweige),
Formatfunktionen, Randfälle (DEMO-PLAYBOOK §12, Verdikte als reine Funktionen)."""

import math

import numpy as np
import pytest

import snk_constants as C
import snk_evaluation as E
import snk_scenario as SC
from snk_newton import loese_gibbs


# ----------------------------------------------------------------------------------------------------------------------------------
# Format und Hilfen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_formatfunktionen_mit_dezimalkomma():
    assert E.de(1.2345, 2) == "1,23" and E.prozent(0.0526, 1) == "5,3 %" and E.de_g(0.0426, 2) == "0,043" and E.de_g(12345.0, 3) == "1,23e4"
    assert E.de_g(0) == "0" and E.de_g(9e-9, 2) == "9e-9"
    assert E.eps_text(C.EPS_H1) == "0,91" and E.eps_text(C.EPS_H3) == "1,8" and E.eps_text(46.0) == "46" and E.eps_text(0.046) == "0,046"


def test_band_und_jsonbar():
    assert E.band([3, 1, 2, 10]) == [2.5, 1.0, 10.0]
    x = E.jsonbar({"a": np.array([1.0, 2.0]), "b": (np.float64(1.5), np.int64(3)), "c": "text", 5: [True]})
    assert x == {"a": [1.0, 2.0], "b": [1.5, 3], "c": "text", "5": [True]} and type(x["b"][1]) is int


def test_settings_aus_zustand_nimmt_die_werte_und_faellt_auf_die_standardwerte_zurueck():
    s = E.settings_aus_zustand({"n_slider": 7, "eps_rel": 46, "k_iter": 5.0, "pfad_t": 0.25})
    assert (s.n, s.eps_rel, s.k_iter, s.pfad_t, s.seed) == (7, 46.0, 5, 0.25, C.DEFAULTS["seed_input"]) and isinstance(s.eps_rel, float)
    assert E.settings_aus_zustand({}) == E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_eps_stufen_enthalten_die_handwerte_und_die_preset_werte():
    for e in (C.EPS_H1, C.EPS_H3, 46.0, 0.46, 0.046):
        assert e in C.EPS_REL_STUFEN
    assert list(C.EPS_REL_STUFEN) == sorted(C.EPS_REL_STUFEN) and C.EPS_REL_STUFEN[0] == 0.01 and C.EPS_REL_STUFEN[-1] == 200.0


# ----------------------------------------------------------------------------------------------------------------------------------
# Referenz, Kennzahlen, Ansicht
# ----------------------------------------------------------------------------------------------------------------------------------

def test_referenz_h1_opt_ecke_und_produktmass():
    i = SC.handbeispiel_h1()
    ref = E.referenz(i)
    assert ref.opt == pytest.approx(0.5, abs=1e-15) and ref.produkt_kosten == pytest.approx(0.75, abs=1e-15) and ref.eintraege_lp == 2
    assert ref.plan_lp == pytest.approx(np.array([[0.5, 0.0], [0.0, 0.5]]), abs=1e-15)


def test_kennzahlen_h1_nach_der_handrechnung():
    i = SC.handbeispiel_h1()
    ref = E.referenz(i)
    P = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])
    kz = E.kennzahlen(i, ref, P)
    assert kz.kosten == pytest.approx(5 / 8, abs=1e-15) and kz.luecke == pytest.approx(0.25, abs=1e-15) and kz.theta == pytest.approx(0.5, abs=1e-15)
    assert kz.tv == pytest.approx(0.25, abs=1e-15) and kz.eintraege == 4 and kz.abstand_lp == pytest.approx(0.5, abs=1e-15)      # vier Zellen mit Abstand 1/8 zur Ecke


def test_ansicht_h1_mit_den_standardwerten_des_presets_gibt_25_prozent_und_einen_durchlauf():
    an = E.ansicht(E.Settings(quelle=C.QUELLE_H1, eps_rel=C.EPS_H1))
    assert an.kz.luecke == pytest.approx(0.25, abs=1e-8) and an.kz.theta == pytest.approx(0.5, abs=1e-7) and an.iterationen == 1 and an.produkte == 2 and an.konvergiert
    assert an.plan == pytest.approx(np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]]), abs=1e-12)


def test_ansicht_h3_nach_einem_durchlauf_roh_minus_20_prozent():
    i = SC.handbeispiel_h3()
    ref = E.referenz(i)
    kk, fehler, luecken = E.rohe_luecke_verlauf(i, ref, C.EPS_H3, (1, 2, 3, 4, 5, 6))
    assert kk == [1, 2, 3, 4, 5, 6] and luecken[0] == pytest.approx(-0.2, abs=1e-14) and fehler[0] == pytest.approx(0.3, abs=1e-14)
    assert luecken[1] == pytest.approx(0.026, abs=5e-4) and luecken[3] == pytest.approx(0.0489, abs=5e-4) and luecken[5] < (3 * math.sqrt(3) - 5) / 4


def test_handbeispiel_luecken_25_prozent_gegen_0_6_prozent():
    h1, h3 = E.handbeispiel_luecken()
    assert h1 == pytest.approx(0.25, abs=1e-10) and h3 == pytest.approx((math.sqrt(1701) - 41) / 40, abs=1e-10) and 0.006 < h3 < 0.0062


def test_ansicht_zufallsinstanz_ist_konsistent_zur_referenz_und_meldet_nicht_konvergiert_beim_limit(monkeypatch):
    an = E.ansicht(E.Settings())
    assert an.konvergiert and an.zeilenfehler < 1e-6 and an.kz.luecke >= 0 and an.kz.eintraege > an.ref.eintraege_lp and an.produkte == 2 * an.iterationen
    monkeypatch.setattr(C, "MAX_ITERATIONEN", 3)
    knapp = E.ansicht(E.Settings(eps_rel=0.01))
    assert not knapp.konvergiert and knapp.iterationen == 3
    kind, text = E.verdikt_ansicht_produkte(knapp, 0.01)
    assert kind == "warning" and "nicht konvergiert" in text


def test_randfaelle_n_gleich_1_gleiche_kosten_und_eps_riesig():
    ein = SC.Instanz(np.array([1.0]), np.array([1.0]), np.array([[1.0]]))
    ref = E.referenz(ein)
    kz = E.kennzahlen(ein, ref, np.array([[1.0]]))
    assert ref.opt == 1.0 and kz.luecke == 0.0 and kz.theta is None and kz.tv == 0.0          # Produktmaß = LP-Optimum: Nenner 0 gibt „keine Angabe“
    gleich = SC.Instanz(np.array([0.5, 0.5]), np.array([0.25, 0.75]), np.full((2, 2), 0.5))
    kz2 = E.kennzahlen(gleich, E.referenz(gleich), np.outer(gleich.a, gleich.b))
    assert kz2.theta is None and kz2.luecke == pytest.approx(0.0, abs=1e-15)
    riesig = E.ansicht(E.Settings(eps_rel=200.0))
    assert riesig.konvergiert and riesig.kz.tv < 0.02


# ----------------------------------------------------------------------------------------------------------------------------------
# Reihen (klein) und Verdikte (handgesetzt)
# ----------------------------------------------------------------------------------------------------------------------------------

def test_reihen_instanzen_sind_fest_und_haengen_nicht_vom_regler_seed_ab():
    paare = E.reihen_instanzen(6, C.KOSTENARTEN[0], 3)
    assert len(paare) == 3
    for index, (inst, ref) in enumerate(paare):
        frisch = SC.make_instanz(6, 6, C.KOSTENARTEN[0], C.SERIEN_SEED, C.ZWECK_REIHE, index)
        assert np.array_equal(inst.kosten, frisch.kosten) and np.array_equal(inst.a, frisch.a) and ref.opt == E.referenz(frisch).opt


def test_reihe_eps_klein_hat_die_erwartete_gestalt_und_ordnung():
    z = E.reihe_eps(6, C.KOSTENARTEN[0], anzahl=4)
    assert [r["eps_rel"] for r in z] == list(C.EPS_REIHE) and all(r["anzahl"] == 4 and r["konvergiert"] == 4 for r in z)
    tv = [r["tv"][0] for r in z]
    assert all(tv[j] < tv[j + 1] for j in range(3)) and all(r["fach"][1] >= 1.0 - 1e-12 for r in z)             # TV wächst mit kleinerem ε, Kosten nie unter OPT
    assert all(r["tv"][1] <= r["tv"][0] <= r["tv"][2] for r in z) and z[0]["produkte"][0] < z[-1]["produkte"][0]


def zeile(e, tv, fach, luecke, eintr, verh, lp_e, n=12, k=30):
    return dict(eps_rel=e, n=n, anzahl=k, konvergiert=k, tv=tv, fach=fach, luecke=luecke, eintraege=eintr, eintraege_verhaeltnis=verh, eintraege_lp=lp_e, produkte=[6, 4, 8])


def test_verdikt_vom_scharfen_zum_weichen_nennt_jede_zahl_und_die_schwelle_fuer_nahe():
    z = [zeile(46.0, [0.0135, 0.004, 0.029], [2.51, 1.48, 4.58], [1.5, 0.5, 3.6], [144, 141, 144], [6.26, 6, 6.3], [23, 23, 23]),
         zeile(0.046, [0.66, 0.5, 0.78], [1.0026, 1.0007, 1.009], [0.00257, 0.0007, 0.009], [40, 25, 57], [1.74, 1.1, 2.5], [23, 23, 23])]
    kind, text = E.verdikt_vom_scharfen_zum_weichen(z)
    assert kind == "info" and "liegt der Plan nahe am Produktmaß (Abstand in Totalvariation 0,013, größter Wert 0,029; n = 12, Median über 30 Instanzen)" in text
    assert "das 2,5-Fache von OPT" in text and "Lücke bei 0,26 %" in text and "das 1,7-Fache der Einträge über 1e-6 der Ecke (40 gegen 23, Median)" in text
    z[0]["tv"] = [0.07, 0.05, 0.1]
    assert "liegt der Plan im Abstand 0,070 vom Produktmaß" in E.verdikt_vom_scharfen_zum_weichen(z)[1]
    assert "enthält die Stufen" in E.verdikt_vom_scharfen_zum_weichen(z[:1])[1]


def skala_zeilen(l20, l40, l80=None):
    def z(n, luecke):
        return dict(n=n, anzahl=30, opt=[0.1, 0.05, 0.2], eps_rel_fest=[0.5 / n * 10, 0.01, 0.9], luecke=luecke, nicht_konvergiert=0, opt_liste=[], eps_rel_liste=[], luecke_liste=[])
    return [z(10, [0.0047, 0.0006, 0.011]), z(20, l20), z(40, l40), z(80, l80 if l80 else [0.0088, 0.006, 0.02])]


def test_verdikt_skala_steigend_gegen_nicht_einheitlich_und_die_handbeispiele():
    kind, text = E.verdikt_skala(skala_zeilen([0.006, 0.003, 0.01], [0.008, 0.004, 0.012], [0.009, 0.005, 0.015]), 0.005)
    assert "ε = 0,005 ist bei n = 10 gleich 0,5 OPT und bei n = 80 gleich 0,062 OPT (Median über 30 Instanzen)" in text
    assert "0,47 % (n = 10), 0,60 % (n = 20), 0,80 % (n = 40), 0,90 % (n = 80); sie steigt in dieser Messung mit n" in text
    assert "25,0 % gegen 0,6 %" in text and "Handbeispielen H1 und H3 mit ε/OPT = 0,91" in text
    kind, text = E.verdikt_skala(skala_zeilen([0.0091, 0.0035, 0.017], [0.0077, 0.0034, 0.0167], [0.0088, 0.0066, 0.0100]), 0.005)
    assert "ist in dieser Messung nicht monoton in n" in text and "bei n = 20 reicht sie zwischen den Instanzen von 0,35 % bis 1,70 %" in text
    assert "Ein Anstieg mit n ist bei 30 Instanzen je n nicht sicher zu sehen" in text and "100 Instanzen je n, quadrierter Abstand, Zufalls-Seed 777) steigt der Median von 0,43 % (n = 10) auf 1,00 % (n = 80)" in text
    kind, text = E.verdikt_skala(skala_zeilen([0.006, 0.003, 0.01], [0.008, 0.004, 0.012], [0.009, 0.005, 0.015]), 0.005)
    assert "Kontrollrechnung" not in text                                                  # steigt die Reihe selbst, braucht es keine Kontrolle


def test_reihe_skala_klein_liefert_je_n_eine_zeile_mit_opt_und_luecke():
    z = E.reihe_skala(0.005, 3, C.KOSTENARTEN[0], ns=(10, 20))
    assert [r["n"] for r in z] == [10, 20] and all(len(r["opt_liste"]) == 3 and r["nicht_konvergiert"] == 0 for r in z)
    assert all(r["eps_rel_fest"][0] == pytest.approx(np.median([0.005 / o for o in r["opt_liste"]]), rel=1e-12) for r in z)
    assert all(0 < r["luecke"][0] < 0.1 for r in z)


def roh_und_prod(neg=29, eps=0.46):
    roh = dict(n=12, anzahl=30, eps_rel=eps, k=[1, 2, 30], zeilenfehler=[[0.41, 0.16, 0.57], [0.24, 0.09, 0.41], [7e-5, 0.0, 0.05]], luecke=[[-0.235, -0.49, 0.012], [-0.09, -0.38, 0.13], [0.119, 0.01, 0.19]],
               negativ=[neg, 23, 0])
    prod = dict(n=12, anzahl=30, zeilen=[dict(eps_rel=e, produkte=[100.0, 50.0, 200.0], nicht_konvergiert=0) for e in C.EPS_PRODUKTE], faktoren=[3.05, 4.04, 2.54, 2.55], eps_faktoren=[3.04, 3.29, 3.04, 3.29])
    return roh, prod


def test_verdikt_produkte_nennt_eine_iteration_die_gewaehlte_zahl_und_die_faktoren():
    roh, prod = roh_und_prod()
    kind, text = E.verdikt_produkte(roh, prod, 30)
    assert text.startswith("Nach 1 Iteration (2 Produkte) liegen die Kosten bei ε/OPT = 0,46 im Median 23,5 % unter dem LP-Optimum, in 29 von 30 Instanzen darunter (n = 12)")
    assert "die Zeilensummen weichen dann im Median um 0,41 (L1)" in text and "Nach 30 Iterationen (60 Produkte) liegen die Kosten bei ε/OPT = 0,46 im Median 11,9 % über dem LP-Optimum" in text
    assert "um den Faktor 2,5 bis 4,0 je Stufe von etwa 3,2 in ε (Median über 30 Instanzen)" in text and "Grenze" not in text
    assert text.count("Nach 1 Iteration") == 1 and "Nach 1 Iterationen" not in E.verdikt_produkte(roh, prod, 1)[1]
    prod["zeilen"][4]["nicht_konvergiert"] = 2
    assert "In 2 Läufen wurde die Grenze von 30.000 Iterationen erreicht." in E.verdikt_produkte(roh, prod, 1)[1]


def test_reihe_rohe_luecke_und_produkte_klein_haben_die_erwartete_gestalt():
    roh = E.reihe_rohe_luecke(6, C.KOSTENARTEN[0], 0.46, ks=(1, 3, 30), anzahl=4)
    assert roh["k"] == [1, 3, 30] and len(roh["luecke"]) == 3 and roh["negativ"][0] >= roh["negativ"][2] and roh["zeilenfehler"][0][0] > roh["zeilenfehler"][2][0]
    prod = E.reihe_produkte(6, C.KOSTENARTEN[0], stufen=(1.4, 0.46, 0.14), anzahl=3)
    assert len(prod["zeilen"]) == 3 and len(prod["faktoren"]) == 2 and prod["faktoren"][0] > 1 and prod["eps_faktoren"][0] == pytest.approx(1.4 / 0.46)


def test_pfad_2x2_lp_springt_einmal_um_2_und_der_weiche_plan_hat_eps_mal_l_gleich_1():
    e, (a, b, C0, C1) = E.pfad_2x2(200, 0.1)
    assert e["lp_wechsel"] == 1 and e["lp_max_sprung"] == pytest.approx(2.0, abs=1e-12) and e["eps_bezug"] == 0.5
    w = e["weich"][0]
    assert w["ok"] and w["eps"] == pytest.approx(0.05) and w["eps_L"] == pytest.approx(1.0, abs=0.01)           # Steigung -1/(4 ε) am Gleichstand
    assert len(e["lp_spruenge"]) == 200 and len(w["spruenge"]) == 200 and max(w["spruenge"]) < 0.11 and e["T"] == 200
    kind, text = E.verdikt_pfad([e], 200, True)
    assert kind == "info" and "springt im Handbeispiel einmal um 2,00 (Höchstwert 2)" in text and "ε·L = 1,00" in text and "klein gegen ε" in text


def test_pfad_2x2_weiche_pfadwerte_gleich_der_geschlossenen_form():
    a = np.array([0.5, 0.5])
    for t in (0.0, 0.25, 0.5, 0.7, 1.0):
        lp, weich = E.pfad_plaene_bei(a, a, np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), t, 0.05)
        assert weich[0, 0] == pytest.approx(0.5 / (1 + math.exp(-(1 - 2 * t) / 0.05)), abs=1e-10)
        if t != 0.5:
            assert lp[0, 0] == (0.5 if t < 0.5 else 0.0)


def test_pfad_20x20_klein_zaehlt_traegerwechsel_und_hat_glatte_weiche_sprunge():
    e = E.pfad_ergebnis(*SC.make_pfad(8, C.KOSTENARTEN[0], 3, 0), 40, (1.0, 0.1), 0.05)
    assert e["lp_wechsel"] >= 3 and 0 < e["lp_max_sprung"] <= 2.0 and len(e["weich"]) == 2 and all(w["ok"] for w in e["weich"])
    sprung = [w["max_sprung"] for w in e["weich"]]
    assert sprung[0] < sprung[1]                    # größeres ε: kleinerer Sprung


def pfad_eintrag(wechsel, sprung, s1, s2, s3):
    return dict(T=800, lp_wechsel=wechsel, lp_max_sprung=sprung, lp_spruenge=[], eps_bezug=0.05,
                weich=[dict(eps_rel=1.0, ok=True, eps=0.05, max_sprung=s1, spruenge=[], eps_L=0.16), dict(eps_rel=0.1, ok=True, eps=0.005, max_sprung=s2, spruenge=[], eps_L=0.08),
                       dict(eps_rel=0.01, ok=True, eps=0.0005, max_sprung=s3, spruenge=[], eps_L=0.03)])


def test_verdikt_pfad_20x20_nennt_wechsel_sprung_und_eps_l():
    kind, text = E.verdikt_pfad([pfad_eintrag(96, 0.19, 0.0019, 0.0098, 0.0345), pfad_eintrag(110, 0.35, 0.0024, 0.0131, 0.0391)], 800)
    assert "mindestens 96 bis 110 Trägerwechsel" in text and "größter Sprung 0,19 bis 0,35 (n = 20, T = 800, 2 Instanzen, Höchstwert 2)" in text
    assert "bei ε/OPT = 1 größter Sprung 0,002 bis 0,002 (ε·L höchstens 0,16)" in text and "bei ε/OPT = 0,01 größter Sprung 0,035 bis 0,039" in text and "klein gegen ε" in text


def verz(kosten, median, steigung, halb, bewertet=(6, 20), eindeutig=20):
    stufen = list(C.EPS_VERZERRUNG)[:len(median)]
    return dict(kosten=kosten, anzahl=20, stufen=stufen, median=median, minimum=median, maximum=median, steigung=steigung, halblog_steigung=halb, eindeutig=eindeutig, nicht_konvergiert=0,
                halblog_besser=bewertet[0], bewertet=bewertet[1])


def test_verdikt_verzerrung_stetig_potenz_und_gitter_exponentiell():
    med = [0.46, 0.21, 0.09, 0.036, 0.014, 0.005, 0.0019, 0.00065, 0.0002, 5.8e-5, 2.2e-5]
    kind, text = E.verdikt_verzerrung(verz(C.VERZERRUNG_KOSTEN[0], med, [1.36, 1.47, 1.59, 1.64, 1.78, 1.69, 1.86, 2.09, 2.13, 1.69], [None] * 10))
    assert "bei 1,4 im ersten Intervall" in text and "bei 1,7 bis 2,1 unter ε/OPT = 0,1" in text and "lokaler Steigung 1,4 bis 2,1" in text and "nicht zu erkennen" in text and "kein exponentieller" not in text and "6 von 20 Instanzen" in text and "20 von 20 Instanzen mit eindeutigem Optimum" in text
    med_g = [0.67, 0.23, 0.057, 0.0095, 0.0007, 8.3e-6, 3.5e-9, 5.6e-13, 8.8e-14, 4e-16, -1e-14]
    steig = [1.87, 2.42, 3.12, 4.58, 7.66, 13.47, None, None, None, None]
    halb = [-0.6, -0.59, -0.58, -0.57, -0.57, -0.56, None, None, None, None]
    kind, text = E.verdikt_verzerrung(verz(C.VERZERRUNG_KOSTEN[1], med_g, steig, halb, (2, 2), 0))
    assert "Gitterkosten (6 × 6, n = 20, 20 Instanzen; 0 von 20 Instanzen mit eindeutigem Optimum)" in text and "(als Anteil von OPT) von 0,67 bei ε/OPT = 1 auf 3,5e-9 bei ε/OPT = 0,032" in text
    assert "wächst von 1,9 auf 13,5" in text and "ln Lücke gegen OPT/ε ist fast linear" in text and "ähnelt einer Exponentialfunktion von 1/ε" in text
    assert "ob die Mehrdeutigkeit des Optimums die Ursache ist, ist nicht getestet" in text
    kind, text = E.verdikt_verzerrung(verz(C.VERZERRUNG_KOSTEN[1], med_g, steig, [-0.6, -0.3, -1.4, -0.57, -0.57, -0.56, None, None, None, None], (2, 2), 0))
    assert "keine Potenz" in text and "Exponentialfunktion" not in text


def test_reihe_verzerrung_klein_stetig_eindeutig_und_gitter_nicht_eindeutig():
    r = E.reihe_verzerrung(C.VERZERRUNG_KOSTEN[0], 4, stufen=(1.0, 0.316, 0.1, 0.0316))
    assert r["eindeutig"] == 4 and r["nicht_konvergiert"] == 0 and all(r["median"][j] > r["median"][j + 1] for j in range(3)) and len(r["steigung"]) == 3 and all(s > 1 for s in r["steigung"])
    g = E.reihe_verzerrung(C.VERZERRUNG_KOSTEN[1], 4, stufen=(1.0, 0.316, 0.1))
    assert g["eindeutig"] == 0 and g["median"][2] < g["median"][1] < g["median"][0] and g["steigung"][1] > g["steigung"][0]            # Gitter: die Steigung wächst


def test_vorgerechnet_schluessel_und_lookup():
    from snk_vorgerechnet import hole, schluessel

    assert schluessel("reihe_eps", n=12, art="a") == "reihe_eps|art='a',n=12"
    assert hole("gibt_es_nicht", n=1) is None
    for name, params, _ in E.vorzurechnen():
        assert hole(name, **params) is not None, name         # alle Standardreihen liegen vorgerechnet in tools/vorgerechnet.json


def test_fehlende_oder_kaputte_vorrechnung_faellt_auf_die_live_rechnung_zurueck(tmp_path, monkeypatch):
    import snk_vorgerechnet as V

    kaputt = tmp_path / "vorgerechnet.json"
    kaputt.write_text("{ kein json", encoding="utf-8")
    for datei in (tmp_path / "gibt_es_nicht.json", kaputt):
        monkeypatch.setattr(V, "DATEI", datei)
        V._lade.cache_clear()
        assert V.hole("reihe_eps", n=6, art=C.KOSTENARTEN[0]) is None
        z = E.eps_v(6, C.KOSTENARTEN[0])                           # rechnet live (n = 6, 30 Instanzen) und gibt dieselbe Gestalt wie die Datei
        assert [r["eps_rel"] for r in z] == list(C.EPS_REIHE) and isinstance(z[0]["tv"], list)
    monkeypatch.undo()
    V._lade.cache_clear()
    assert V.hole("reihe_eps", n=12, art=C.KOSTENARTEN[0]) is not None
