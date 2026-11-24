"""Messreihen der README (alle Zahlen, die die README nennt): rechnet mit dem Code der Demo (Zufalls-Seed 2026 der Reihen, SplitMix64) und schreibt `tools/messreihen.json`.

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/messreihen.py      (rund 90 Sekunden)
Die Reihen für die Standardeinstellungen entsprechen `tools/vorgerechnet.json`; hier kommen n = 40, die zweite Kostenart und die Handbeispiele dazu.
`tests/test_claims.py` bildet die README-Zahlen aus dieser Datei und rechnet kleine Reihen neu (als Band)."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import snk_constants as C  # noqa: E402
import snk_evaluation as E  # noqa: E402

QUAD, ABST = C.KOSTENARTEN
GROSSE_LISTEN = ("spruenge", "lp_spruenge", "opt_liste", "eps_rel_liste", "luecke_liste")


def kurz(x):
    """Entfernt die langen Listen (Sprünge je Schritt, Werte je Instanz), die die README nicht braucht."""
    if isinstance(x, dict):
        return {k: kurz(v) for k, v in x.items() if k not in GROSSE_LISTEN}
    if isinstance(x, list):
        return [kurz(v) for v in x]
    return x


def zeit(name, f):
    t = time.time()
    r = f()
    print(f"{name}: {time.time() - t:.1f} s", flush=True)
    return kurz(E.jsonbar(r))


def main():
    d = {"meta": {"seed": C.SERIEN_SEED, "anzahl": C.ANZAHL_REIHE, "max_iter": C.MAX_ITERATIONEN, "tol": C.TOL}}
    d["eps_n12"] = zeit("eps n=12", lambda: E.reihe_eps(12, QUAD))
    d["eps_n20"] = zeit("eps n=20", lambda: E.reihe_eps(20, QUAD))
    d["eps_n40"] = zeit("eps n=40", lambda: E.reihe_eps(40, QUAD))
    d["eps_n12_abstand"] = zeit("eps n=12 Abstand", lambda: E.reihe_eps(12, ABST))
    d["skala"] = zeit("skala", lambda: E.reihe_skala(0.005, 30, QUAD))
    d["skala_kontrolle_2026"] = zeit("skala Kontrolle 100 Instanzen, Zufalls-Seed 2026", lambda: E.reihe_skala(0.005, C.ANZAHL_KONTROLLE, QUAD))
    d["skala_kontrolle_777"] = zeit("skala Kontrolle 100 Instanzen, Zufalls-Seed 777", lambda: E.reihe_skala(0.005, C.ANZAHL_KONTROLLE, QUAD, seed=C.KONTROLL_SEED))
    d["roh_n12"] = zeit("roh n=12", lambda: E.reihe_rohe_luecke(12, QUAD, 0.46))
    d["roh_n40"] = zeit("roh n=40", lambda: E.reihe_rohe_luecke(40, QUAD, 0.46))
    d["produkte_n12"] = zeit("produkte n=12", lambda: E.reihe_produkte(12, QUAD))
    d["produkte_n40"] = zeit("produkte n=40", lambda: E.reihe_produkte(40, QUAD))
    d["pfad"] = {str(T): zeit(f"pfad T={T}", lambda T=T: E.pfad_reihe(C.N_PFAD, QUAD, C.DEFAULTS["seed_input"], T)) for T in C.PFAD_T_STUFEN}
    d["pfad_2x2"] = {str(T): zeit(f"pfad 2x2 T={T}", lambda T=T: E.pfad_2x2(T, 0.1)[0]) for T in C.PFAD_T_STUFEN}
    d["verzerrung_stetig"] = zeit("verzerrung stetig", lambda: E.reihe_verzerrung(C.VERZERRUNG_KOSTEN[0], 20))
    d["verzerrung_gitter"] = zeit("verzerrung gitter", lambda: E.reihe_verzerrung(C.VERZERRUNG_KOSTEN[1], 20))
    d["hand_luecken"] = list(E.handbeispiel_luecken())
    (ROOT / "tools" / "messreihen.json").write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben: tools/messreihen.json")


if __name__ == "__main__":
    main()
