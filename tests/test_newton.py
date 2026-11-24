"""Newton-Verfahren auf dem Dual (snk_newton): Konvergenz bis ε/OPT = 0,003, Randfehler, Fortsetzung in ε, Abbruchmeldung, Randfälle.
Der Plan selbst wird in tests/test_oracle_sinkhorn.py gegen die Iteration und gegen L-BFGS auf dem Dual geprüft."""

import numpy as np
import pytest

import snk_newton as N
import snk_scenario as SC
from snk_evaluation import referenz
from snk_sinkhorn import loese_sinkhorn, plan_aus_potentialen


def inst(k, n=12, art="quadrierter Abstand"):
    i = SC.make_instanz(n, n, art, 3, 9, k)
    return i, referenz(i)


@pytest.mark.parametrize("eps_rel", [46.0, 1.0, 0.1, 0.01, 0.003])
def test_konvergiert_bis_zum_kleinsten_eps_mit_randfehler_unter_1e_12(eps_rel):
    for k in range(6):
        i, ref = inst(k)
        erg = N.loese_gibbs(i.a, i.b, i.kosten, eps_rel * ref.opt)
        assert erg.ok and erg.fehler < 1e-12, (k, erg.fehler)
        assert erg.plan.sum(axis=1) == pytest.approx(i.a, abs=1e-12) and erg.plan.sum(axis=0) == pytest.approx(i.b, abs=1e-12)
        assert (erg.plan >= 0).all() and erg.plan.shape == i.kosten.shape


def test_kosten_nahe_am_lp_optimum_fuer_kleines_eps():
    i, ref = inst(0)
    erg = N.loese_gibbs(i.a, i.b, i.kosten, 0.003 * ref.opt)
    k = float((erg.plan * i.kosten).sum())
    assert ref.opt <= k + 1e-12 and (k - ref.opt) / ref.opt < 1e-3


def test_warmstart_aus_der_nachbarloesung_braucht_wenige_schritte():
    i, ref = inst(1)
    e1, e2 = 0.1 * ref.opt, 0.09 * ref.opt
    erg1 = N.loese_gibbs(i.a, i.b, i.kosten, e1)
    warm = N.loese_gibbs_newton(i.a, i.b, i.kosten, e2, erg1.f, erg1.g)
    assert warm.ok and warm.schritte <= 6 and warm.plan == pytest.approx(N.loese_gibbs(i.a, i.b, i.kosten, e2).plan, abs=1e-9)


def test_zu_wenige_schritte_melden_nicht_konvergiert_statt_einen_plan_vorzutaeuschen():
    i, ref = inst(2)
    erg = N.loese_gibbs_newton(i.a, i.b, i.kosten, 0.01 * ref.opt, max_schritte=1)
    assert not erg.ok and erg.fehler > 1e-12


def test_pfad_mit_fortsetzung_gibt_je_eps_ein_ergebnis_und_fallende_kosten():
    i, ref = inst(3)
    eps_liste = [3 * ref.opt] + [e * ref.opt for e in (1.0, 0.3, 0.1, 0.03)]
    erg = N.loese_gibbs_pfad(i.a, i.b, i.kosten, eps_liste)
    assert [e for e, _ in erg] == eps_liste and all(r.ok for _, r in erg)
    kosten = [float((r.plan * i.kosten).sum()) for _, r in erg]
    assert all(kosten[j] >= kosten[j + 1] - 1e-12 for j in range(len(kosten) - 1))            # die Kosten fallen mit ε


def test_gleiche_kosten_geben_das_produktmass():
    a, b = np.array([0.1, 0.3, 0.6]), np.array([0.5, 0.5])
    erg = N.loese_gibbs(a, b, np.full((3, 2), 0.4), 0.05)
    assert erg.ok and erg.plan == pytest.approx(np.outer(a, b), abs=1e-10)


def test_unmoegliches_eps_endet_ohne_ausnahme():
    i, ref = inst(4)
    erg = N.loese_gibbs(i.a, i.b, i.kosten, 1e-9 * ref.opt)
    assert isinstance(erg.ok, bool)
    if erg.ok:
        assert erg.fehler < 1e-12


def test_newton_und_iteration_haben_denselben_plan_und_potentiale_bis_auf_die_konstante():
    i, ref = inst(5, n=8)
    eps = 0.5 * ref.opt
    nw = N.loese_gibbs(i.a, i.b, i.kosten, eps)
    sk = loese_sinkhorn(i.a, i.b, i.kosten, eps, tol=1e-13)
    assert plan_aus_potentialen(i.a, i.b, i.kosten, eps, sk.f, sk.g) == pytest.approx(nw.plan, abs=1e-9)
    d = nw.f - sk.f
    assert d == pytest.approx(np.full_like(d, d[0]), abs=1e-7)       # f nur bis auf eine Konstante bestimmt (f + t, g - t gibt denselben Plan)
