"""Zeilen der README-Tabellen, aus `tools/messreihen.json` mit denselben Formatfunktionen wie in der App gebildet (snk_evaluation.de, de_g, prozent).

Die README muss jede dieser Zeilen wörtlich enthalten (`tests/test_claims.py`); `python tests/readme_zeilen.py` druckt sie zum Einfügen."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import snk_constants as C  # noqa: E402
import snk_evaluation as E  # noqa: E402

J = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))


def bt(b, stellen=3) -> str:
    """Median [Minimum; Maximum]."""
    return f"{E.de_g(b[0], stellen)} [{E.de_g(b[1], stellen)}; {E.de_g(b[2], stellen)}]"


def zeile_eps(r: dict) -> str:
    return (f"| {E.eps_text(r['eps_rel'])} | {bt(r['tv'])} | {bt(r['fach'])} | {E.prozent(r['luecke'][0], 2)} | {E.band_ganz(r['eintraege'])} ({E.ganz(r['eintraege_lp'][0])}) | {E.band_ganz(r['produkte'])} |")


def zeile_skala(r: dict) -> str:
    return f"| {r['n']} | {bt(r['opt'])} | {bt(r['eps_rel_fest'])} | {bt([100 * w for w in r['luecke']])} % |"


def zeile_roh(r: dict, i: int) -> str:
    return (f"| {r['k'][i]} | {bt([100 * w for w in r['luecke'][i]], 3)} % | {r['negativ'][i]} von {r['anzahl']} | {E.de_g(r['zeilenfehler'][i][0], 2)} |")


def zeile_produkte(z: dict, faktor, eps_faktor, anzahl: int) -> str:
    f = "–" if faktor is None else f"{E.de(faktor, 2)} (ε-Faktor {E.de(eps_faktor, 2)})"
    return f"| {E.eps_text(z['eps_rel'])} | {E.band_ganz(z['produkte'])} | {z['nicht_konvergiert']} von {anzahl} | {f} |"


def zeilen_produkte(p: dict) -> list:
    z = p["zeilen"]
    return [zeile_produkte(z[j], None if j == 0 else p["faktoren"][j - 1], None if j == 0 else p["eps_faktoren"][j - 1], p["anzahl"]) for j in range(len(z))]


def zeile_pfad(T: int, er: float) -> str:
    ws = [w for e in J["pfad"][str(T)] for w in e["weich"] if abs(w["eps_rel"] - er) < 1e-12 and w["ok"]]
    return f"| {T} | {E.eps_text(er)} | {bt(E.band([w['max_sprung'] for w in ws]))} | {bt(E.band([w['eps_L'] for w in ws]))} |"


def zeile_verzerrung(r: dict, j: int) -> str:
    s = r["steigung"][j] if j < len(r["steigung"]) else None
    return f"| {E.eps_text(r['stufen'][j])} | {E.de_g(r['median'][j], 3)} | {E.de_g(r['minimum'][j], 2)} bis {E.de_g(r['maximum'][j], 2)} | {E.de(s, 2) if s is not None else '–'} |"


def verzerrung_aufgeloest(r: dict) -> list:
    return [j for j in range(len(r["stufen"])) if r["median"][j] == r["median"][j] and r["median"][j] > E.LUECKE_BODEN]


if __name__ == "__main__":
    for key in ("eps_n12", "eps_n40"):
        print("##", key)
        for r in J[key]:
            print(zeile_eps(r))
    print("## skala")
    for r in J["skala"]:
        print(zeile_skala(r))
    for key in ("roh_n12", "roh_n40"):
        print("##", key)
        r = J[key]
        for i, k in enumerate(r["k"]):
            if k in (1, 3, 10, 30, 1000):
                print(zeile_roh(r, i))
    for key in ("produkte_n12", "produkte_n40"):
        print("##", key)
        for z in zeilen_produkte(J[key]):
            print(z)
    print("## pfad")
    for T in (50, 200, 800):
        for er in E.PFAD_EPS_REL:
            print(zeile_pfad(T, er))
    for key in ("verzerrung_stetig", "verzerrung_gitter"):
        print("##", key)
        r = J[key]
        for j in verzerrung_aufgeloest(r):
            print(zeile_verzerrung(r, j))
