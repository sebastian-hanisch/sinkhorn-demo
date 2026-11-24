"""Instanzen, Handbeispiele, Pfade und Kopien: gleiche Eingabe gleiche Instanz (auch in einem zweiten Prozess), Normierung max C = 1, Summen 1, Gitterinstanzen,
Kopien der Hilfsmodule gegen ihre Quellen (DEMO-PLAYBOOK §4: jede Kopie eines Kerns braucht einen Kopier-Test)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import snk_constants as C
import snk_scenario as SC

ROOT = Path(__file__).resolve().parent.parent
ART = C.KOSTENARTEN[0]


def test_make_instanz_formen_summen_und_normierung():
    for art in C.KOSTENARTEN:
        i = SC.make_instanz(7, 9, art, 3, 1, 4)
        assert i.kosten.shape == (7, 9) and i.lager.shape == (7, 2) and i.kunden.shape == (9, 2)
        assert i.a.sum() == pytest.approx(1.0, abs=1e-14) and i.b.sum() == pytest.approx(1.0, abs=1e-14) and (i.a > 0).all() and (i.b > 0).all()
        assert i.kosten.max() == 1.0 and i.kosten.min() >= 0.0


def test_kosten_sind_quadrierter_abstand_bzw_abstand_geteilt_durch_den_groessten_wert():
    i = SC.make_instanz(4, 5, ART, 8, 1, 0)
    q = ((i.lager[:, None, :] - i.kunden[None, :, :]) ** 2).sum(-1)
    assert i.kosten == pytest.approx(q / q.max(), rel=1e-14)
    j = SC.make_instanz(4, 5, C.KOSTENARTEN[1], 8, 1, 0)
    assert j.kosten == pytest.approx(np.sqrt(q) / np.sqrt(q).max(), rel=1e-14) and np.array_equal(i.lager, j.lager)      # gleiche Orte, andere Kosten
    with pytest.raises(ValueError):
        SC.kosten_aus_orten(i.lager, i.kunden, "unbekannt")


def test_massen_dirichlet_und_mischung():
    g = SC.instanz_rng(1, 1, 1)
    assert SC.massen(g, 5, 0.0) == pytest.approx(np.full(5, 0.2), abs=1e-15)
    m = SC.massen(SC.instanz_rng(1, 1, 2), 50, 0.5)
    assert m.sum() == pytest.approx(1.0, abs=1e-14) and m.min() >= 0.5 / 50                      # Boden (1 - s)/n
    mittel = np.mean([SC.massen(SC.instanz_rng(2, 3, k), 10, 1.0).max() for k in range(200)])
    assert 0.2 < mittel < 0.4                                                                    # Dirichlet(1) mit n = 10: das größte Gewicht liegt bei etwa 0,29


def test_gleicher_seed_gleiche_instanz_und_verschiedene_zwecke_und_indizes_sind_verschieden():
    a = SC.make_instanz(6, 6, ART, 5, 1, 0)
    assert np.array_equal(a.kosten, SC.make_instanz(6, 6, ART, 5, 1, 0).kosten)
    for andere in (SC.make_instanz(6, 6, ART, 6, 1, 0), SC.make_instanz(6, 6, ART, 5, 2, 0), SC.make_instanz(6, 6, ART, 5, 1, 1)):
        assert not np.array_equal(a.kosten, andere.kosten)


def test_zweiter_prozess_mit_anderem_hashseed_rechnet_dieselbe_instanz_und_denselben_plan():
    code = ("import sys; sys.path.insert(0, r'%s'); sys.dont_write_bytecode = True\n"
            "import snk_scenario as SC, snk_evaluation as E\n"
            "i = SC.make_instanz(9, 9, 'quadrierter Abstand', 5, 1, 0); r = E.referenz(i)\n"
            "z = E.reihe_eps(6, 'quadrierter Abstand', stufen=(1.0,), anzahl=3)\n"
            "print(repr((float(i.kosten.sum()), r.opt, z)))" % str(ROOT))
    ausgaben = []
    for hashseed in ("1", "2"):
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": hashseed, "PATH": "", "SYSTEMROOT": "C:\\Windows", "OMP_NUM_THREADS": "1"}, timeout=120)
        assert r.returncode == 0, r.stderr
        ausgaben.append(r.stdout.strip())
    assert ausgaben[0] == ausgaben[1] and ausgaben[0]


def test_gitterinstanz_hat_verschiedene_zellen_massen_eins_durch_n_und_ganzzahlige_l1_kosten():
    for k in range(10):
        i = SC.make_gitter_instanz(20, 6, 5, C.ZWECK_VERZERRUNG, k)
        assert len({tuple(p) for p in i.lager}) == 20 and len({tuple(p) for p in i.kunden}) == 20
        assert i.a == pytest.approx(np.full(20, 0.05), abs=1e-15) and i.kosten.max() == 1.0
        roh = i.kosten * np.abs(i.lager[:, None, :] - i.kunden[None, :, :]).sum(-1).max()
        assert roh == pytest.approx(np.round(roh), abs=1e-12)                                    # L1-Abstände auf dem Gitter sind ganze Zahlen
    with pytest.raises(ValueError):
        SC.make_gitter_instanz(40, 6, 5, 1, 0)


def test_handbeispiele_h1_und_h3():
    h1, h3 = SC.handbeispiel_h1(), SC.handbeispiel_h3()
    assert h1.kosten.tolist() == [[0.5, 1.0], [1.0, 0.5]] and h1.a.tolist() == [0.5, 0.5] and h1.b.tolist() == [0.5, 0.5]
    assert h3.kosten.tolist() == [[0.0, 1.0], [1.0, 0.0]] and h3.a.tolist() == [0.75, 0.25] and h3.b.tolist() == [0.25, 0.75]
    assert h1.lager.shape == (2, 2) and h3.kunden.shape == (2, 2) and h1.name == C.QUELLE_H1 and h3.name == C.QUELLE_H3


def test_pfad_hat_zwei_unabhaengige_kostenmatrizen_max_eins_und_massen_mit_boden():
    a, b, C0, C1 = SC.make_pfad(20, ART, 35, 0)
    assert C0.shape == C1.shape == (20, 20) and C0.max() == C1.max() == 1.0 and not np.array_equal(C0, C1)
    assert a.sum() == pytest.approx(1.0, abs=1e-14) and a.min() >= (1 - C.S_PFAD) / 20 and b.min() >= (1 - C.S_PFAD) / 20
    assert np.array_equal(SC.make_pfad(20, ART, 35, 0)[2], C0) and not np.array_equal(SC.make_pfad(20, ART, 35, 1)[2], C0)


# ----------------------------------------------------------------------------------------------------------------------------------
# Kopien
# ----------------------------------------------------------------------------------------------------------------------------------

def test_rng_ist_eine_bytegleiche_kopie_der_vorlage_im_website_repo_falls_vorhanden():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / "snk_rng.py").read_bytes() == vorlage.read_bytes()


def test_simplex_und_eindeutigkeitstest_sind_kopien_der_quelle_in_zuordnen_transportieren_demo_falls_vorhanden():
    quelle = ROOT.parent / "zuordnen-transportieren-demo"
    if not (quelle / "ztr_simplex.py").exists():
        pytest.skip("Quelle nicht vorhanden (Demo allein ausgecheckt)")
    for name in ("simplex", "eindeutigkeit"):
        original = (quelle / f"ztr_{name}.py").read_text(encoding="utf-8").replace("ztr_", "snk_")
        assert (ROOT / f"snk_{name}.py").read_text(encoding="utf-8") == original, name


def test_splitmix64_bekannte_ausgaben_fuer_seed_0():
    from snk_rng import SplitMix64

    g = SplitMix64(0)
    assert [g.next_u64() for _ in range(4)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]
