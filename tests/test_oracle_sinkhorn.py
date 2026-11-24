"""Orakel für die Sinkhorn-Iteration, das Newton-Verfahren auf dem Dual und das LP-Optimum: jeweils ein ANDERER Rechenweg als der der Demo.

- Weicher Plan: (a) das Dual unabhängig mit `scipy.optimize.minimize` (L-BFGS-B, eigener Gradient) gelöst; (b) die naive Iteration im K-Bereich (u, v, K = exp(-C/ε), wie im Lehrbuch) liefert
  nach jedem k denselben Plan wie die Log-Domäne; (c) die KL-Projektion von K auf die Pläne mit Nelder-Mead auf den freien Einträgen (n = m = 3); (d) die Gibbs-Form P = diag(u) K diag(v) und die Randsummen.
- LP-Optimum (ε/OPT hängt daran): `linprog` (HiGHS, Simplex und innere Punkte), `networkx.network_simplex` und der Mindestkostenfluss von OR-Tools auf ganzzahligen Instanzen.
- Eindeutigkeit des LP-Plans (Experiment 5 zählt damit): gegen ein zweites LP auf der optimalen Fläche.
- 2 × 2-Formel x(t) = σ((1 − 2t)/ε)/2 (Handbeispiel H4).
Welche Ecke ein Löser bei Gleichstand wählt, wird nie verglichen: geprüft wird Wert, Zulässigkeit und Eindeutigkeit. Alle Läufe zusammen unter 10 Sekunden."""

import math

import numpy as np
import pytest

import snk_constants as C
import snk_newton as N
import snk_scenario as SC
import snk_simplex as S
from orakel_zahlen import ORAKEL
from snk_eindeutigkeit import ist_eindeutig
from snk_sinkhorn import loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale

sopt = pytest.importorskip("scipy.optimize")


def instanz(k, n=None, m=None):
    g = SC.instanz_rng(11, 12, k)
    n = n or 3 + g.below(4)
    m = m or 3 + g.below(4)
    a = np.array([g.expovariate(1.0) + 0.05 for _ in range(n)])
    b = np.array([g.expovariate(1.0) + 0.05 for _ in range(m)])
    Cm = np.array([[g.random() for _ in range(m)] for _ in range(n)])
    return a / a.sum(), b / b.sum(), Cm / Cm.max()


def lp(a, b, Cm, methode="highs-ds"):
    n, m = Cm.shape
    A = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    skala = 1e6                                         # HiGHS meldet bei Massen unter etwa 1e-7 „infeasible“: Massen skalieren
    r = sopt.linprog(Cm.ravel(), A_eq=A, b_eq=np.concatenate([a, b]) * skala, bounds=(0, None), method=methode)
    assert r.status == 0, r.message
    return r.fun / skala


# ----------------------------------------------------------------------------------------------------------------------------------
# Weicher Plan
# ----------------------------------------------------------------------------------------------------------------------------------

def dual_plan_lbfgs(a, b, Cm, eps):
    """Plan aus dem Dual Phi(f, g) = a·f + b·g − eps Σ a_i b_j exp((f_i + g_j − C_ij)/eps), maximiert mit L-BFGS-B (eigene Rechnung, nicht der Code der Demo)."""
    n, m = Cm.shape

    def plan(x):
        return np.outer(a, b) * np.exp((x[:n, None] + x[None, n:] - Cm) / eps)

    def fg(x):
        P = plan(x)
        wert = a @ x[:n] + b @ x[n:] - eps * P.sum()
        grad = np.concatenate([a - P.sum(axis=1), b - P.sum(axis=0)])
        return -wert, -grad

    r = sopt.minimize(fg, np.zeros(n + m), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-16, gtol=1e-13, maxcor=30))
    return plan(r.x)


@pytest.mark.parametrize("eps_rel", [0.5, 2.0])
def test_plan_gleich_dual_mit_lbfgs_auf_zufallsinstanzen(eps_rel):
    for k in range(ORAKEL["dual"]):
        a, b, Cm = instanz(k)
        eps = eps_rel * lp(a, b, Cm)
        erg = loese_sinkhorn(a, b, Cm, eps, tol=1e-12, max_iter=30000)
        P = plan_aus_potentialen(a, b, Cm, eps, erg.f, erg.g)
        assert erg.konvergiert, k
        assert P == pytest.approx(dual_plan_lbfgs(a, b, Cm, eps), abs=2e-7), (k, eps_rel)


def naive_k_bereich(a, b, Cm, eps, k):
    """Die Lehrbuch-Iteration u = a / (K v), v = b / (K^T u) mit K = exp(-C/eps), Start u = v = 1; Plan nach k Durchläufen."""
    K = np.exp(-Cm / eps)
    u, v = np.ones(len(a)), np.ones(len(b))
    for _ in range(k):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return u[:, None] * K * v[None, :]


def test_log_domaene_gleich_naive_iteration_im_k_bereich_nach_jedem_k():
    for k_inst in range(ORAKEL["naiv"]):
        a, b, Cm = instanz(k_inst)
        eps = 0.8 * lp(a, b, Cm)
        for k, f, g in verlauf_potentiale(a, b, Cm, eps, [1, 2, 3, 5, 8, 20, 60]):
            assert plan_aus_potentialen(a, b, Cm, eps, f, g) == pytest.approx(naive_k_bereich(a, b, Cm, eps, k), rel=1e-9, abs=1e-13), (k_inst, k)


def test_gibbs_form_und_randsummen_des_konvergierten_plans():
    for k in range(10):
        a, b, Cm = instanz(k)
        eps = 0.6 * lp(a, b, Cm)
        erg = loese_sinkhorn(a, b, Cm, eps, tol=1e-13)
        P = plan_aus_potentialen(a, b, Cm, eps, erg.f, erg.g)
        u, v = a * np.exp(erg.f / eps), b * np.exp(erg.g / eps)
        assert P == pytest.approx(u[:, None] * np.exp(-Cm / eps) * v[None, :], rel=1e-12)       # P = diag(u) K diag(v)
        assert P.sum(axis=1) == pytest.approx(a, abs=1e-12) and P.sum(axis=0) == pytest.approx(b, abs=1e-12)
        assert (P > 0).all()                                                                       # dicht: kein Eintrag ist null


def test_kl_projektion_von_k_auf_die_kopplungen_mit_nelder_mead_n_gleich_3():
    """Die KL-Projektion von K = exp(-C/ε) auf die Pläne mit den Randsummen a und b (Benamou u. a. 2015, Gl. 6), unabhängig gelöst: die vier freien Einträge des oberen
    linken 2 × 2-Blocks, der Rest folgt aus den Randsummen; Nelder-Mead auf dem verallgemeinerten KL (SLSQP und trust-constr konvergierten hier nicht)."""
    for k in range(ORAKEL["kl"]):
        a, b, Cm = instanz(k, 3, 3)
        eps = 0.7 * lp(a, b, Cm)
        K = np.exp(-Cm / eps)

        def voll(z):
            Q = np.zeros((3, 3))
            Q[:2, :2] = z.reshape(2, 2)
            Q[:2, 2] = a[:2] - Q[:2, :2].sum(axis=1)
            Q[2, :2] = b[:2] - Q[:2, :2].sum(axis=0)
            Q[2, 2] = a[2] - Q[2, :2].sum()
            return Q

        def kl(z):
            Q = voll(z)
            return 1e10 if (Q <= 0).any() else float((Q * np.log(Q / K) - Q + K).sum())

        r = sopt.minimize(kl, np.outer(a, b)[:2, :2].ravel(), method="Nelder-Mead", options=dict(xatol=1e-13, fatol=1e-16, maxiter=20000, maxfev=20000))
        erg = loese_sinkhorn(a, b, Cm, eps, tol=1e-13)
        assert plan_aus_potentialen(a, b, Cm, eps, erg.f, erg.g) == pytest.approx(voll(r.x), abs=1e-6), k


def test_newton_plan_gleich_sinkhorn_plan_und_gleich_dual_lbfgs():
    for k in range(10):
        a, b, Cm = instanz(k)
        eps = 0.4 * lp(a, b, Cm)
        nw = N.loese_gibbs(a, b, Cm, eps)
        erg = loese_sinkhorn(a, b, Cm, eps, tol=1e-12)
        assert nw.ok and nw.plan == pytest.approx(plan_aus_potentialen(a, b, Cm, eps, erg.f, erg.g), abs=1e-8), k
        assert nw.plan == pytest.approx(dual_plan_lbfgs(a, b, Cm, eps), abs=2e-7), k


def test_zwei_mal_zwei_formel_x_von_t():
    a = np.array([0.5, 0.5])
    for t, eps in ((0.3, 0.1), (0.5, 0.1), (0.62, 0.05), (0.5, 0.01), (0.45, 0.2)):
        Cm = np.array([[t, 1 - t], [1 - t, t]])
        nw = N.loese_gibbs(a, a, Cm, eps)
        assert nw.ok and nw.plan[0, 0] == pytest.approx(0.5 / (1 + math.exp(-(1 - 2 * t) / eps)), abs=1e-12), (t, eps)


# ----------------------------------------------------------------------------------------------------------------------------------
# LP-Optimum und Eindeutigkeit
# ----------------------------------------------------------------------------------------------------------------------------------

def test_optimalwert_gleich_linprog_highs_mit_n_ungleich_m():
    for k in range(ORAKEL["lp"]):
        inst = SC.make_instanz(2 + k % 11, 2 + (k * 3) % 11, C.KOSTENARTEN[k % 2], 7, 3, k)
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        assert lo.kosten == pytest.approx(lp(inst.a, inst.b, inst.kosten), rel=1e-9, abs=1e-12), k
        assert not S.pruefe_loesung(inst.a, inst.b, inst.kosten, lo)


def test_optimalwert_gleich_innere_punkte_von_highs():
    for k in range(0, 3 * ORAKEL["ipm"], 3):
        inst = SC.make_instanz(8, 8, C.KOSTENARTEN[0], 5, 3, k)
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        assert lo.kosten == pytest.approx(lp(inst.a, inst.b, inst.kosten, "highs-ipm"), rel=1e-7, abs=1e-9)


def test_ganzzahlige_instanzen_gleich_networkx_und_ortools_mit_abweichung_0():
    nx = pytest.importorskip("networkx")
    mcf_modul = pytest.importorskip("ortools.graph.python.min_cost_flow")
    for k in range(ORAKEL["netz"]):
        g = SC.instanz_rng(5, 6, k)
        n, m = 2 + g.below(5), 2 + g.below(5)
        a = [g.below(6) + 1 for _ in range(n)]
        gesamt = sum(a)
        schnitte = sorted(g.below(gesamt + 1) for _ in range(m - 1))
        b = [y - x for x, y in zip([0] + schnitte, schnitte + [gesamt])]
        Cm = np.array([[g.below(20) for _ in range(m)] for _ in range(n)], float)
        lo = S.loese_transport(a, b, Cm, "steilste")
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(("L", i), demand=-a[i])
        for j in range(m):
            G.add_node(("K", j), demand=b[j])
        for i in range(n):
            for j in range(m):
                G.add_edge(("L", i), ("K", j), weight=int(Cm[i, j]))
        assert nx.network_simplex(G)[0] == lo.kosten
        mcf = mcf_modul.SimpleMinCostFlow()
        for i in range(n):
            for j in range(m):
                mcf.add_arc_with_capacity_and_unit_cost(i, n + j, gesamt, int(Cm[i, j]))
        for i in range(n):
            mcf.set_node_supply(i, a[i])
        for j in range(m):
            mcf.set_node_supply(n + j, -b[j])
        assert mcf.solve() == mcf.OPTIMAL and mcf.optimal_cost() == lo.kosten, k


def test_eindeutigkeit_gegen_ein_zweites_lp_auf_der_optimalen_flaeche():
    """Der Plan ist eindeutig genau dann, wenn auf der optimalen Fläche jede Zelle nur einen Wert annimmt (min und max je Zelle mit linprog)."""
    n = 4
    A = np.vstack([np.kron(np.eye(n), np.ones((1, n))), np.kron(np.ones((1, n)), np.eye(n))])
    eindeutig, mehrere = 0, 0
    for k in range(ORAKEL["eind"]):
        g = SC.instanz_rng(9, 4, k)
        Cm = np.array([[float(g.below(3)) for _ in range(n)] for _ in range(n)])
        a = b = np.full(n, 1.0 / n)
        lo = S.loese_transport(a, b, Cm, "steilste")
        wert = lp(a, b, Cm)
        A_eq = np.vstack([A, Cm.ravel()[None, :]])
        b_eq = np.concatenate([a, b, [wert]])
        spanne = 0.0
        for z in range(n * n):
            c = np.zeros(n * n)
            c[z] = 1.0
            unten = sopt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
            oben = sopt.linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
            assert unten.status == 0 and oben.status == 0
            spanne = max(spanne, -oben.fun - unten.fun)
        erwartet = spanne < 1e-9
        assert ist_eindeutig(Cm, lo.plan, lo.u, lo.v) == erwartet, k
        eindeutig += erwartet
        mehrere += not erwartet
    assert eindeutig >= 3 and mehrere >= 3          # beide Fälle kommen vor
