"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine
Grenzen und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen. Dazu: je Preset die erwartete Aussage, Permalink, Randfälle, Worst-Case am Regler-Maximum.

Laufzeit: Die Reihen sind für die Standardeinstellungen vorgerechnet (tools/vorgerechnet.json); jede andere Einstellung rechnet live (Zwischenspeicher gilt im Prozess).
Die Tests wählen darum eine kleine Ausgangslage (Adresszeile: n = 6, 10 Instanzen, T = 50), wo es nur um das Zusammenspiel der Regler geht."""

import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import snk_constants as C
import snk_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio")
KLEIN = dict(n="6", i2="10", i5="5", T="50")     # Adresszeile der kleinen Ausgangslage


def _app(**adresse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in adresse.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _klein() -> AppTest:
    return _app(**KLEIN)


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _setze(at, **werte):
    for key, wert in werte.items():
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (werte, [str(e) for e in at.exception])
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in ("success", "info", "warning", "markdown", "caption") for e in getattr(at, kind))


def _metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_all_sections():
    at = _app()
    titel = " ".join(s.value for s in at.subheader)
    for abschnitt in ("Experiment 1: Vom scharfen zum weichen Plan", "Experiment 2: Die Skala", "Experiment 3: Wie viele Produkte", "Experiment 4: Glatt statt springend",
                      "Experiment 5: Wie schnell verschwindet die Verzerrung?", "Wo die Annahmen enden"):
        assert abschnitt in titel, abschnitt
    assert [e.label for e in at.expander if "Mathematische Formulierung" in e.label] == ["📐 Mathematische Formulierung"]
    assert any("Annahmen, Grenzen und bekannte Probleme" in e.label for e in at.expander)
    assert len(at.metric) == 6 and len(at.info) >= 8 and not at.warning
    assert len({p.key for p in at.get("plotly_chart")}) == len(at.get("plotly_chart")) >= 10         # jedes Diagramm hat einen eigenen Schlüssel


def test_every_button_click_does_not_raise():
    at = _klein()
    labels = [b.label for b in at.button]
    assert len(labels) == len(C.PRESETS) + 1, labels                      # sieben Presets und der Zufalls-Seed-Button
    for index, label in enumerate(labels):
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    at = _klein()
    schluessel = [s.key for s in at.sidebar.slider]
    assert schluessel == ["n_slider", "pfad_t"], schluessel
    for key in schluessel:
        for edge in ("min", "max"):
            regler = _regler(at, key)
            wert = regler.min if edge == "min" else regler.max
            regler.set_value(int(wert) if isinstance(regler.value, int) else float(wert)).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------------------------
# Auswahlregler und Presets
# ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,optionen", [
    ("quelle", C.QUELLEN),
    ("kosten_art", C.KOSTENARTEN),
    ("eps_rel", (C.EPS_REL_STUFEN[0], C.EPS_H1, C.EPS_H3, C.EPS_REL_STUFEN[-1])),
    ("eps_roh", (C.EPS_ROH_STUFEN[0], C.EPS_ROH_STUFEN[-1])),
    ("inst2", C.INSTANZEN2_STUFEN[1:]),
    ("k_iter", (C.K_STUFEN[0], 3, C.K_STUFEN[-1])),
    ("pfad_art", C.PFAD_ARTEN),
    ("pfad_T", C.PFAD_T_STUFEN),
    ("verz_kosten", C.VERZERRUNG_KOSTEN),
    ("inst5", C.INSTANZEN5_STUFEN[1:]),
])
def test_optionen_der_auswahlregler_laufen_ohne_fehler(key, optionen):
    at = _klein()
    for option in optionen:
        _setze(at, **{key: option})


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler_und_zeigt_seine_einstellung(name):
    at = _klein()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, wert in C.PRESETS[name].items():
        assert at.session_state[key] == wert, (name, key)


def test_preset_von_hand_zeigt_25_prozent_luecke_zwei_produkte_und_vier_eintraege():
    at = _klein()
    at.sidebar.button(key="preset_Von Hand: 2 × 2").click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert m["Kostenlücke (⟨C, P⟩ − OPT) / OPT"] == "25,00 %" and m["Einträge über 1e-6"] == "4" and m["Matrix-Vektor-Produkte bis Zeilenfehler 1e-8"] == "2"
    assert m["Anteil θ am Weg LP → Produktmaß"] == "0,5" and m["Abstand zum Produktmaß (Totalvariation)"] == "0,25"
    assert "die Ecke des LP hat 2" in " ".join(str(x.delta) for x in at.metric)
    assert "Handbeispiel H1: 2 × 2, gleiche Massen: 2 Lager, 2 Kunden" in _texte(at) and "ε/OPT = 0,91" in _texte(at)


def test_preset_fast_produktmass_zeigt_kleine_tv_und_viele_eintraege():
    at = _klein()
    at.sidebar.button(key="preset_Fast Produktmaß (ε/OPT 46)").click().run(timeout=TIMEOUT)
    assert not at.exception
    tv = float(_metrics(at)["Abstand zum Produktmaß (Totalvariation)"].replace(",", "."))
    assert tv < 0.06 and "Bei ε/OPT = 46 hat der Plan 144 Einträge über 1e-6 (die Ecke des LP 23)" in _texte(at)


def test_preset_fast_scharf_hat_kleine_luecke_aber_mehr_eintraege_als_die_ecke():
    at = _klein()
    at.sidebar.button(key="preset_Fast scharf (0,046)").click().run(timeout=TIMEOUT)
    assert not at.exception
    m = _metrics(at)
    luecke = float(m["Kostenlücke (⟨C, P⟩ − OPT) / OPT"].replace(" %", "").replace(",", "."))
    assert luecke < 2.0 and int(m["Einträge über 1e-6"]) > 23


def test_preset_abbruch_nach_1_iteration_nennt_negative_rohe_luecke_und_k_1():
    at = _klein()
    at.sidebar.button(key="preset_Abbruch nach 1 Iteration").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["k_iter"] == 1
    text = _texte(at)
    assert re.search(r"Nach 1 Iteration \(2 Produkte\) liegen die Kosten bei ε/OPT = 0,46 im Median [\d,]+ % unter dem LP-Optimum, in \d+ von 30 Instanzen darunter", text)
    assert "**Nach k = 1 Iterationen (2 Produkte):** rohe Kostenlücke im Median -" in text.replace("−", "-")


def test_preset_pfad_gleichstand_zeigt_den_sprung_2_und_eps_l_um_1():
    at = _klein()
    at.sidebar.button(key="preset_Pfad: Gleichstand").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at)
    assert "springt im Handbeispiel einmal um 2,00 (Höchstwert 2)" in text and "Handbeispiel 2 × 2, C(t) = [[t, 1 − t], [1 − t, t]]" in text
    assert re.search(r"ε·L = (0,9\d|1,00)", text)


def test_preset_gitterkosten_zeigt_nicht_eindeutige_optima_und_steigende_steigung():
    at = _klein()
    at.sidebar.button(key="preset_Gitterkosten").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at)
    assert "Gitterkosten (6 × 6, n = 20, 5 Instanzen; 0 von 5 Instanzen mit eindeutigem Optimum)" in text or "Instanzen mit eindeutigem Optimum" in text
    assert "die Steigung der Median-Kurve wächst von" in text


# ----------------------------------------------------------------------------------------------------------------------------------
# Permalink, Randfälle, Worst-Case, Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_permalink_uebernimmt_werte_begrenzt_und_eingerastet():
    at = _app(n="9999", eps="0.5", k="9999", q="unbekannt", t="0.333", T="49", i2="10", i5="5")
    s = at.session_state
    assert s["n_slider"] == C.N_MAX and s["eps_rel"] == 0.46 and s["k_iter"] == 1000 and s["quelle"] == C.DEFAULTS["quelle"] and s["pfad_t"] == 0.33 and s["pfad_T"] == 50


def test_adresszeile_spiegelt_die_einstellungen():
    at = _setze(_klein(), n_slider=7, eps_rel=4.6)
    assert at.query_params["n"] in (["7"], "7") and at.query_params["eps"] in (["4.6"], "4.6") and at.query_params["i2"] in (["10"], "10")


def test_randfaelle_alle_minima_und_seed_0():
    at = _klein()
    _setze(at, quelle=C.QUELLE_ZUFALL, n_slider=C.N_MIN, eps_rel=C.EPS_REL_STUFEN[-1], seed_input=0, eps_roh=C.EPS_ROH_STUFEN[0], inst2=C.INSTANZEN2_STUFEN[0],
           k_iter=1, pfad_art=C.PFAD_ARTEN[0], pfad_T=50, pfad_t=0.0, verz_kosten=C.VERZERRUNG_KOSTEN[1], inst5=C.INSTANZEN5_STUFEN[0])


def test_worst_case_alle_maxima_zugleich_ohne_fehler():
    """Regler-Maximum: n = 20, ε/OPT = 0,01 (bis 30 000 Iterationen), 30 Instanzen, k = 1000, 20 × 20-Pfade mit T = 800, 20 Instanzen, Pfadwert 1. Die Zeit wird nicht geprüft (Abschnitt 1c)."""
    at = _app()
    _setze(at, n_slider=C.N_MAX, eps_rel=C.EPS_REL_STUFEN[0], eps_roh=C.EPS_ROH_STUFEN[-1], inst2=C.INSTANZEN2_STUFEN[-1], k_iter=C.K_STUFEN[-1], pfad_art=C.PFAD_ARTEN[1],
           pfad_T=C.PFAD_T_STUFEN[-1], pfad_t=1.0, verz_kosten=C.VERZERRUNG_KOSTEN[0], inst5=C.INSTANZEN5_STUFEN[-1])
    assert len(at.metric) == 6


def test_die_ansicht_nennt_keine_planbegriffe_und_keine_umgebungsaussagen():
    at = _app()
    gesamt = _texte(at) + " " + " ".join(e.label for e in at.expander) + " " + " ".join(s.value for s in at.subheader)
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Orakel", "Saatwert", "Sinkhorn-Distanz", "Sinkhorn-Divergenz", "wasserstein", "Stück"):
        assert verboten not in gesamt, verboten
    assert not re.search(r"\btr[äa]gt\b|\btragen\b", gesamt)
    assert "Zufalls-Seed" in " ".join(w.label for w in at.sidebar.number_input)


def test_jeder_regler_der_seitenleiste_hat_einen_hilfetext():
    at = _app()
    for art in REGLER_ARTEN:
        for w in getattr(at.sidebar, art):
            if w.key in P.SETTING_SPECS and w.key != "seed_input":
                assert w.help, w.key
