"""Eigenschaftstests des weichen Plans (ORACLE-PLAYBOOK §5): Skalierung, Symmetrien, Grenzfälle, Schranken. Je 10 bis 30 Instanzen, feste Zufalls-Seeds."""

import numpy as np
import pytest

import snk_constants as C
import snk_newton as N
import snk_scenario as SC
from snk_evaluation import kennzahlen, referenz
from snk_sinkhorn import loese_sinkhorn, plan_aus_potentialen


def inst(k, n=8):
    i = SC.make_instanz(n, n + k % 3, C.KOSTENARTEN[k % 2], 21, 8, k)
    return i, referenz(i)


def plan(i, eps, tol=1e-12):
    erg = loese_sinkhorn(i.a, i.b, i.kosten, eps, tol=tol, max_iter=30000)
    assert erg.konvergiert
    return plan_aus_potentialen(i.a, i.b, i.kosten, eps, erg.f, erg.g)


def entropie(w):
    return float(-(w * np.log(w)).sum())


def test_kosten_und_eps_mit_demselben_faktor_geben_denselben_plan():
    for k in range(15):
        i, ref = inst(k)
        eps = 0.5 * ref.opt
        P = plan(i, eps)
        for faktor in (0.1, 7.0):
            erg = loese_sinkhorn(i.a, i.b, faktor * i.kosten, faktor * eps, tol=1e-12)
            assert plan_aus_potentialen(i.a, i.b, faktor * i.kosten, faktor * eps, erg.f, erg.g) == pytest.approx(P, abs=1e-10)


def test_umnummerieren_der_lager_und_kunden_permutiert_den_plan():
    for k in range(12):
        i, ref = inst(k)
        g = SC.instanz_rng(1, 2, k)
        zeilen = list(range(i.n))
        spalten = list(range(i.m))
        g.shuffle(zeilen)
        g.shuffle(spalten)
        eps = 0.4 * ref.opt
        P = plan(i, eps)
        Pq = loese_sinkhorn(i.a[zeilen], i.b[spalten], i.kosten[np.ix_(zeilen, spalten)], eps, tol=1e-12)
        Q = plan_aus_potentialen(i.a[zeilen], i.b[spalten], i.kosten[np.ix_(zeilen, spalten)], eps, Pq.f, Pq.g)
        assert Q == pytest.approx(P[np.ix_(zeilen, spalten)], abs=1e-10)


def test_vertauschen_von_angebot_und_nachfrage_transponiert_den_plan():
    for k in range(12):
        i, ref = inst(k)
        eps = 0.4 * ref.opt
        P = plan(i, eps)
        erg = loese_sinkhorn(i.b, i.a, i.kosten.T, eps, tol=1e-12)
        assert plan_aus_potentialen(i.b, i.a, i.kosten.T, eps, erg.f, erg.g) == pytest.approx(P.T, abs=1e-10)


def test_eps_gegen_unendlich_gibt_das_produktmass():
    for k in range(10):
        i, ref = inst(k)
        P = plan(i, 1e4 * ref.opt, tol=1e-13)
        assert 0.5 * np.abs(P - np.outer(i.a, i.b)).sum() < 1e-3


def test_kostenluecke_liegt_zwischen_null_und_eps_mal_der_summe_der_randentropien():
    """0 <= <C, P_eps> - OPT <= eps (h(a) + h(b)): P_eps minimiert <C, P> - eps H(P), H(P) <= h(a) + h(b), H >= 0 (eigene Herleitung)."""
    for k in range(20):
        i, ref = inst(k)
        for e in (0.05, 0.5, 5.0):
            eps = e * ref.opt
            kosten = float((plan(i, eps) * i.kosten).sum())
            assert -1e-12 <= kosten - ref.opt <= eps * (entropie(i.a) + entropie(i.b)) + 1e-12, (k, e)


def test_kosten_des_weichen_plans_wachsen_mit_eps_je_instanz():
    for k in range(12):
        i, ref = inst(k)
        eps_liste = [0.01 * ref.opt * 2.5 ** j for j in range(9)]
        pfad = N.loese_gibbs_pfad(i.a, i.b, i.kosten, eps_liste[::-1])
        kosten = [float((r.plan * i.kosten).sum()) for _, r in pfad][::-1]
        assert all(kosten[j] <= kosten[j + 1] + 1e-12 for j in range(len(kosten) - 1)), k


def test_der_plan_ist_dicht_und_hat_mindestens_so_viele_eintraege_wie_die_ecke():
    for k in range(15):
        i, ref = inst(k)
        P = plan(i, 0.1 * ref.opt)
        assert kennzahlen(i, ref, P).eintraege >= ref.eintraege_lp and (P > 0).all()


def test_tv_zum_produktmass_ist_klein_bei_grossem_eps_und_waechst_mit_kleinerem_eps():
    for k in range(10):
        i, ref = inst(k)
        tv = [kennzahlen(i, ref, plan(i, e * ref.opt)).tv for e in (200.0, 20.0, 2.0, 0.2)]
        assert tv[0] < 0.02 and all(tv[j] <= tv[j + 1] + 1e-12 for j in range(3)), (k, tv)


def test_theta_liegt_zwischen_0_und_1_und_faellt_mit_kleinerem_eps():
    for k in range(10):
        i, ref = inst(k)
        th = [kennzahlen(i, ref, plan(i, e * ref.opt)).theta for e in (50.0, 5.0, 0.5, 0.05)]
        assert all(0.0 <= t <= 1.0 for t in th) and all(th[j] >= th[j + 1] - 1e-12 for j in range(3)), (k, th)


def test_gleicher_zufalls_seed_gleiche_instanz():
    a1 = SC.make_instanz(9, 9, C.KOSTENARTEN[0], 5, 1, 0)
    a2 = SC.make_instanz(9, 9, C.KOSTENARTEN[0], 5, 1, 0)
    assert np.array_equal(a1.kosten, a2.kosten) and np.array_equal(a1.a, a2.a)
    assert not np.array_equal(a1.kosten, SC.make_instanz(9, 9, C.KOSTENARTEN[0], 6, 1, 0).kosten)
