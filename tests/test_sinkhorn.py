"""Einheiten der Sinkhorn-Iteration gegen die Handrechnungen H1 bis H6 (Brüche, `fractions`) und gegen Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5).

Die Handbeispiele sind so klein, dass jede Zahl von Hand nachzurechnen ist; die Rechenwege stehen im Expander „Mathematische Formulierung“ der App."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import snk_constants as C
from snk_sinkhorn import (
    aktualisiere_f,
    aktualisiere_g,
    lse_zeilen,
    loese_sinkhorn,
    plan_aus_potentialen,
    pruefe_massen,
    verlauf_potentiale,
    zeilenfehler,
)

A2 = np.array([0.5, 0.5])
C2 = np.array([[0.5, 1.0], [1.0, 0.5]])
EPS_H1 = 1.0 / (2.0 * math.log(3.0))
A3, B3 = np.array([0.75, 0.25]), np.array([0.25, 0.75])
C3 = np.array([[0.0, 1.0], [1.0, 0.0]])
EPS_H3 = 1.0 / math.log(3.0)


def plan_nach(a, b, Cm, eps, k):
    """Der Plan nach k Durchläufen (k >= 1) aus der Iteration der Demo."""
    f, g = verlauf_potentiale(a, b, Cm, eps, [k])[0][1:]
    return plan_aus_potentialen(a, b, Cm, eps, f, g)


def kosten(P, Cm):
    return float((P * Cm).sum())


# ----------------------------------------------------------------------------------------------------------------------------------
# Bausteine
# ----------------------------------------------------------------------------------------------------------------------------------

def test_lse_zeilen_gegen_direkte_summe_und_bei_grossen_betraegen():
    M = np.array([[0.0, math.log(3.0)], [-2.0, 1.0]])
    assert lse_zeilen(M) == pytest.approx([math.log(4.0), math.log(math.exp(-2.0) + math.e)], rel=1e-14)
    gross = np.array([[1000.0, 1000.0], [-1000.0, -1000.0 + math.log(2.0)]])
    assert lse_zeilen(gross) == pytest.approx([1000.0 + math.log(2.0), -1000.0 + math.log(3.0)], rel=1e-14)       # die naive Summe liefe über bzw. unter


def test_f_und_g_halbschritt_als_formel_von_hand():
    # H1, g = 0: f_i = -eps ln sum_j b_j exp(-C_ij / eps) = -eps ln (1/2 (1/3 + 1/9)) = -eps ln (2/9)
    lb = np.log(A2)
    f = aktualisiere_f(lb, np.zeros(2), C2, EPS_H1)
    assert f == pytest.approx([-EPS_H1 * math.log(2.0 / 9.0)] * 2, rel=1e-14)
    g = aktualisiere_g(np.log(A2), np.zeros(2), np.ascontiguousarray(C2.T), EPS_H1)
    assert g == pytest.approx(f, rel=1e-14)                                    # symmetrische Instanz


def test_zeilenfehler_ist_der_l1_abstand_der_zeilensummen():
    P = np.array([[0.2, 0.1], [0.3, 0.3]])
    assert zeilenfehler(P, np.array([0.25, 0.65])) == pytest.approx(0.05 + 0.05)


def test_plan_aus_potentialen_hat_gibbs_form_und_potentiale_null_geben_das_produktmass_mal_kern():
    P = plan_aus_potentialen(A2, A2, C2, EPS_H1, np.zeros(2), np.zeros(2))
    assert P == pytest.approx(np.outer(A2, A2) * np.exp(-C2 / EPS_H1), rel=1e-14)


def test_pruefe_massen_lehnt_unpassende_eingaben_ab():
    ok = (A2, A2, C2)
    pruefe_massen(*ok)
    with pytest.raises(ValueError):
        pruefe_massen(np.array([0.0, 1.0]), A2, C2)          # Masse 0: die Log-Domäne braucht positive Massen
    with pytest.raises(ValueError):
        pruefe_massen(np.array([0.5, 0.6]), A2, C2)          # verschiedene Gesamtmassen
    with pytest.raises(ValueError):
        pruefe_massen(A2, A2, np.ones((3, 2)))               # Form passt nicht
    with pytest.raises(ValueError):
        loese_sinkhorn(A2, A2, C2, 0.0)                      # eps muss positiv sein


# ----------------------------------------------------------------------------------------------------------------------------------
# H1: zwei Lager, zwei Kunden, gleiche Massen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_h1_kern_und_ein_durchlauf_in_brueche():
    K = np.exp(-C2 / EPS_H1)
    assert K == pytest.approx(np.array([[1 / 3, 1 / 9], [1 / 9, 1 / 3]]), abs=1e-15, rel=0)
    # von Hand: u = a / (K 1) = (1/2) / (1/3 + 1/9) = 9/8; K^T u = (1/2, 1/2) = b, also v = 1; P = 9/8 K
    Kf = [[Fr(1, 3), Fr(1, 9)], [Fr(1, 9), Fr(1, 3)]]
    u = Fr(1, 2) / (Kf[0][0] + Kf[0][1])
    assert u == Fr(9, 8)
    Pf = [[u * Kf[i][j] for j in range(2)] for i in range(2)]
    assert Pf == [[Fr(3, 8), Fr(1, 8)], [Fr(1, 8), Fr(3, 8)]]
    P = plan_nach(A2, A2, C2, EPS_H1, 1)
    assert P == pytest.approx(np.array(Pf, dtype=float), abs=1e-15)
    assert zeilenfehler(P, A2) < 1e-15 and P.sum(axis=0) == pytest.approx(A2, abs=1e-15)


def test_h1_kosten_luecke_theta_tv_und_eintraege_in_brueche():
    Pf = [[Fr(3, 8), Fr(1, 8)], [Fr(1, 8), Fr(3, 8)]]
    Cf = [[Fr(1, 2), Fr(1)], [Fr(1), Fr(1, 2)]]
    k = sum(Pf[i][j] * Cf[i][j] for i in range(2) for j in range(2))
    assert k == Fr(5, 8) and (k - Fr(1, 2)) / Fr(1, 2) == Fr(1, 4)                       # Lücke 25 %
    assert (k - Fr(1, 2)) / (Fr(3, 4) - Fr(1, 2)) == Fr(1, 2)                           # Produktmaß kostet 3/4: theta = 1/2
    assert Fr(1, 2) * sum(abs(Pf[i][j] - Fr(1, 4)) for i in range(2) for j in range(2)) == Fr(1, 4)    # TV zum Produktmaß
    assert all(Pf[i][j] > 0 for i in range(2) for j in range(2))                         # dicht: vier Einträge, die Ecke hat zwei


@pytest.mark.parametrize("e", [0.05, 0.2, 1.0, 5.0, 50.0])
def test_h1_luecke_geschlossen_als_funktion_von_eps_durch_opt(e):
    P = plan_nach(A2, A2, C2, e * 0.5, 1)
    assert (kosten(P, C2) - 0.5) / 0.5 == pytest.approx(1.0 / (1.0 + math.exp(1.0 / e)), abs=1e-12)


def test_h1_grenzen_produktmass_und_scharfer_plan():
    assert 1.0 / (1.0 + math.exp(1.0 / 1e6)) == pytest.approx(0.5, abs=1e-6)         # eps -> unendlich: Kosten 3/4 = Produktmaß
    assert 1.0 / (1.0 + math.exp(1.0 / 0.005)) < 1e-80                                 # eps/OPT = 0,005: Lücke unter 1e-80


# ----------------------------------------------------------------------------------------------------------------------------------
# H2: ε hat die Einheit der Kosten
# ----------------------------------------------------------------------------------------------------------------------------------

def test_h2_kosten_und_eps_mal_10_geben_denselben_plan_nur_kosten_mal_10_nicht():
    P = plan_nach(A2, A2, C2, EPS_H1, 1)
    assert plan_nach(A2, A2, 10 * C2, 10 * EPS_H1, 1) == pytest.approx(P, abs=1e-14)
    q = Fr(3 ** 10, 1 + 3 ** 10)
    assert 1 - q == Fr(1, 59050)
    P10 = plan_nach(A2, A2, 10 * C2, EPS_H1, 1)
    assert (kosten(P10, 10 * C2) - 5.0) / 5.0 == pytest.approx(1 / 59050, rel=1e-6)


# ----------------------------------------------------------------------------------------------------------------------------------
# H3: die rohe Kostenlücke kann negativ sein
# ----------------------------------------------------------------------------------------------------------------------------------

def test_h3_ein_durchlauf_in_brueche_und_rohe_luecke_minus_20_prozent():
    K3 = [[Fr(1), Fr(1, 3)], [Fr(1, 3), Fr(1)]]
    A, B = [Fr(3, 4), Fr(1, 4)], [Fr(1, 4), Fr(3, 4)]
    u = [A[i] / (K3[i][0] + K3[i][1]) for i in range(2)]
    assert u == [Fr(9, 16), Fr(3, 16)]
    Ktu = [K3[0][j] * u[0] + K3[1][j] * u[1] for j in range(2)]
    assert Ktu == [Fr(5, 8), Fr(3, 8)]
    v = [B[j] / Ktu[j] for j in range(2)]
    assert v == [Fr(2, 5), Fr(2)]
    P1 = [[u[i] * K3[i][j] * v[j] for j in range(2)] for i in range(2)]
    assert P1 == [[Fr(9, 40), Fr(3, 8)], [Fr(1, 40), Fr(3, 8)]]
    Cf = [[Fr(0), Fr(1)], [Fr(1), Fr(0)]]
    k1 = sum(P1[i][j] * Cf[i][j] for i in range(2) for j in range(2))
    assert k1 == Fr(2, 5) and (k1 - Fr(1, 2)) / Fr(1, 2) == Fr(-1, 5)                    # rohe Lücke -20 %
    assert [P1[0][0] + P1[0][1], P1[1][0] + P1[1][1]] == [Fr(3, 5), Fr(2, 5)]            # Zeilensummen (3/5, 2/5) statt (3/4, 1/4)
    assert abs(Fr(3, 5) - A[0]) + abs(Fr(2, 5) - A[1]) == Fr(3, 10)                      # Zeilenfehler 3/10
    P = plan_nach(A3, B3, C3, EPS_H3, 1)
    assert P == pytest.approx(np.array(P1, dtype=float), abs=1e-14)
    assert zeilenfehler(P, A3) == pytest.approx(0.3, abs=1e-14) and P.sum(axis=0) == pytest.approx(B3, abs=1e-14)
    assert (kosten(P, C3) - 0.5) / 0.5 == pytest.approx(-0.2, abs=1e-14)


def test_h3_grenzplan_und_die_rohe_luecke_steigt_von_unten_gegen_den_grenzwert():
    x = (9 - 3 * math.sqrt(3)) / 16
    assert 8 * x ** 2 - 9 * x + 27 / 16 == pytest.approx(0, abs=1e-14)                     # Gibbs-Bedingung x^2 = 9 (3/4 - x)(1/4 - x)
    P = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=1e-14, max_iter=2000)
    Pg = plan_aus_potentialen(A3, B3, C3, EPS_H3, P.f, P.g)
    assert Pg[0, 0] == pytest.approx(x, abs=1e-12) and P.konvergiert
    grenze = (kosten(Pg, C3) - 0.5) / 0.5
    assert grenze == pytest.approx((3 * math.sqrt(3) - 5) / 4, abs=1e-12)                  # +4,9 %
    roh = [(kosten(plan_nach(A3, B3, C3, EPS_H3, k), C3) - 0.5) / 0.5 for k in range(1, 7)]
    assert roh[0] < 0 < roh[1] and all(roh[i] < roh[i + 1] < grenze + 1e-12 for i in range(5))


def test_h3_dasselbe_eps_wie_h1_gibt_hier_die_luecke_0_6_prozent_statt_25():
    P = loese_sinkhorn(A3, B3, C3, EPS_H1, tol=1e-14, max_iter=5000)
    Pg = plan_aus_potentialen(A3, B3, C3, EPS_H1, P.f, P.g)
    assert (kosten(Pg, C3) - 0.5) / 0.5 == pytest.approx((math.sqrt(1701) - 41) / 40, abs=1e-10)
    assert EPS_H1 / 0.5 == pytest.approx(1 / math.log(3), abs=1e-15)                        # ε/OPT = 0,910 in beiden Instanzen
    assert C.EPS_H1 == pytest.approx(1 / math.log(3), abs=1e-15) and C.EPS_H3 == pytest.approx(2 / math.log(3), abs=1e-15)


# ----------------------------------------------------------------------------------------------------------------------------------
# Zählung und Abbruch
# ----------------------------------------------------------------------------------------------------------------------------------

def test_h6_sieben_iterationen_sind_vierzehn_produkte():
    erg = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=0.0, max_iter=7)
    assert erg.iterationen == 7 and erg.produkte == 14 and not erg.konvergiert
    assert 2 * 60 * 60 * 1000 == 7_200_000          # Multiplikationen 2 n^2 k bei n = 60, k = 1000


def test_abbruch_bei_erreichter_genauigkeit_und_der_gemeldete_fehler_ist_der_zeilenfehler_des_plans():
    erg = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=1e-8)
    assert erg.konvergiert and erg.zeilenfehler < 1e-8 and erg.iterationen > 1
    P = plan_aus_potentialen(A3, B3, C3, EPS_H3, erg.f, erg.g)
    assert zeilenfehler(P, A3) == pytest.approx(erg.zeilenfehler, rel=1e-6, abs=1e-12)


def test_max_iter_null_und_nicht_konvergiert_beim_limit():
    erg = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=1e-8, max_iter=0)
    assert erg.iterationen == 0 and not erg.konvergiert
    knapp = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=1e-15, max_iter=3)
    assert knapp.iterationen == 3 and not knapp.konvergiert and knapp.zeilenfehler > 1e-15


def test_warmstart_aus_der_loesung_braucht_keine_iteration():
    erg = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=1e-10)
    neu = loese_sinkhorn(A3, B3, C3, EPS_H3, tol=1e-8, f0=erg.f, g0=erg.g)
    assert neu.iterationen == 0 and neu.konvergiert


def test_verlauf_potentiale_gleich_einzelnen_laeufen_ohne_abbruch():
    rng = np.random.default_rng(3)
    a, b = rng.random(5) + 0.1, rng.random(5) + 0.1
    a, b = a / a.sum(), b / b.sum()
    Cm = rng.random((5, 5))
    ks = [1, 2, 5, 17]
    for k, f, g in verlauf_potentiale(a, b, Cm, 0.2, ks):
        erg = loese_sinkhorn(a, b, Cm, 0.2, tol=0.0, max_iter=k)
        assert erg.iterationen == k and erg.f == pytest.approx(f, abs=1e-14) and erg.g == pytest.approx(g, abs=1e-14)
    with pytest.raises(ValueError):
        verlauf_potentiale(a, b, Cm, 0.2, [0, 3])


# ----------------------------------------------------------------------------------------------------------------------------------
# Randfälle
# ----------------------------------------------------------------------------------------------------------------------------------

def test_n_gleich_1_ist_das_produktmass_mit_luecke_null():
    a, b, Cm = np.array([1.0]), np.array([1.0]), np.array([[0.7]])
    erg = loese_sinkhorn(a, b, Cm, 0.1)
    P = plan_aus_potentialen(a, b, Cm, 0.1, erg.f, erg.g)
    assert P.shape == (1, 1) and P[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_gleiche_kosten_geben_das_produktmass_fuer_jedes_eps():
    a, b = np.array([0.2, 0.3, 0.5]), np.array([0.6, 0.4])
    for eps in (1.0, 0.1, 0.001):
        erg = loese_sinkhorn(a, b, np.full((3, 2), 0.5), eps)
        assert plan_aus_potentialen(a, b, np.full((3, 2), 0.5), eps, erg.f, erg.g) == pytest.approx(np.outer(a, b), abs=1e-7)


def test_sehr_kleines_eps_ueberlaeuft_nicht_in_der_log_domaene():
    rng = np.random.default_rng(5)
    a = rng.random(6) + 0.1
    b = rng.random(6) + 0.1
    a, b = a / a.sum(), b / b.sum()
    Cm = rng.random((6, 6))
    erg = loese_sinkhorn(a, b, Cm, 1e-4, tol=1e-6, max_iter=3000)             # exp(-C/eps) wäre exp(-10000) = 0 in der K-Domäne
    P = plan_aus_potentialen(a, b, Cm, 1e-4, erg.f, erg.g)
    assert np.isfinite(P).all() and P.sum() == pytest.approx(1.0, abs=1e-6)


def test_abbruchtest_kostet_ein_produkt_mehr_als_gemeldet(monkeypatch):
    """Gezählte Log-Sum-Exp-Aufrufe: bei gemeldeten 2k Produkten werden 2k + 1 ausgewertet (der f-Schritt des Abbruchtests bleibt beim Abbruch übrig; README, Zählweise)."""
    import snk_sinkhorn as S

    aufrufe = []
    echt = S.lse_zeilen
    monkeypatch.setattr(S, "lse_zeilen", lambda M: (aufrufe.append(1), echt(M))[1])
    rng = np.random.default_rng(3)
    P12 = rng.random((12, 12))
    fall = [(A2, A2, C2, EPS_H1, {}), (A3, B3, C3, EPS_H3, {}), (A3, B3, C3, EPS_H3, dict(tol=0.0, max_iter=7)),
            (np.full(12, 1 / 12), np.full(12, 1 / 12), P12 / P12.max(), 0.3, {})]
    for a, b, Cm, eps, kw in fall:
        aufrufe.clear()
        erg = loese_sinkhorn(a, b, Cm, eps, **kw)
        assert erg.produkte >= 2 and len(aufrufe) == erg.produkte + 1, (erg.produkte, len(aufrufe))
    # Warmstart, sofort konvergiert: null Iterationen, ein Auswertungsschritt des Tests
    erg = loese_sinkhorn(A3, B3, C3, EPS_H3)
    aufrufe.clear()
    erg2 = loese_sinkhorn(A3, B3, C3, EPS_H3, f0=erg.f, g0=erg.g)
    assert erg2.produkte == 0 and len(aufrufe) == 1
