"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzen und einrasten; jedes Preset lässt sich rechnen."""

import pytest

import snk_constants as C
import snk_evaluation as E
import snk_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.SETTING_SPECS), name          # jedes Preset legt ALLE Regler fest
        assert C.PRESET_HELP[name]
    assert list(C.PRESETS) == ["Von Hand: 2 × 2", "Fast Produktmaß (ε/OPT 46)", "Weich (0,46)", "Fast scharf (0,046)", "Abbruch nach 1 Iteration", "Pfad: Gleichstand", "Gitterkosten"]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_no_two_presets_are_identical_and_the_default_is_exactly_one_preset():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)
    assert [n for n, p in C.PRESETS.items() if p == C.DEFAULTS] == ["Weich (0,46)"]
    assert E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"q", "kosten", "n", "eps", "seed", "er", "i2", "k", "pfad", "T", "t", "vk", "i5"}      # keine doppelte Adressparameter


def test_preset_hilfetexte_nennen_keine_messzahlen():
    """Die Hilfetexte der Presets nennen keine gemessenen Zahlen (jede Zahl der App steht in der README oder in einem Test); erlaubt sind nur Reglerwerte und Handwerte."""
    import re

    erlaubt = ("1e-6", "2 × 2", "1/ln 3", "t = 0,5", "Experiment 4", "Experiment 5")
    for name, text in C.PRESET_HELP.items():
        rest = text
        for token in erlaubt:
            rest = rest.replace(token, "")
        assert "%" not in rest and not re.search(r"\d", rest), (name, rest)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laesst_sich_rechnen_und_zeigt_die_erwartete_einstellung(name):
    s = E.settings_aus_zustand(C.PRESETS[name])
    an = E.ansicht(s)
    assert an.plan.shape == (an.inst.n, an.inst.m) and an.ref.opt > 0
    if name == "Von Hand: 2 × 2":
        assert an.kz.luecke == pytest.approx(0.25, abs=1e-8) and an.inst.name == C.QUELLE_H1
    if name == "Abbruch nach 1 Iteration":
        assert s.k_iter == 1


def test_einrasten_und_begrenzen_des_permalinks():
    sp = P.SETTING_SPECS
    assert P._eingerastet(sp["n_slider"], 500) == C.N_MAX and P._eingerastet(sp["n_slider"], -3) == C.N_MIN
    assert P._eingerastet(sp["eps_rel"], 0.5) == 0.46 and P._eingerastet(sp["eps_rel"], 1e9) == 200.0 and P._eingerastet(sp["eps_rel"], C.EPS_H1) == C.EPS_H1
    assert P._eingerastet(sp["k_iter"], 4) in (3, 5) and P._eingerastet(sp["k_iter"], 9999) == 1000
    assert P._eingerastet(sp["pfad_t"], 0.333) == 0.33 and P._eingerastet(sp["pfad_t"], 7.0) == 1.0 and P._eingerastet(sp["pfad_t"], -1.0) == 0.0
    assert P._eingerastet(sp["quelle"], C.QUELLE_H3) == C.QUELLE_H3 and P._eingerastet(sp["quelle"], "unbekannt") is None
    assert P._eingerastet(sp["pfad_T"], 100) in (50, 200) and P._eingerastet(sp["inst2"], 25) in (20, 30)


def test_hilfetext_fast_scharf_sagt_nur_mehr_eintraege_als_die_ecke_und_die_ansicht_belegt_es():
    """Prüferbefund: „deutlich mehr“ war für die Instanz des Presets (30 gegen 23) zu stark; der Text sagt „noch mehr“, und die Instanz hat tatsächlich mehr Einträge als die Ecke bei kleiner Lücke."""
    text = C.PRESET_HELP["Fast scharf (0,046)"]
    assert "noch mehr Einträge über 1e-6 als die Ecke des LP" in text and "deutlich" not in text
    an = E.ansicht(E.settings_aus_zustand(C.PRESETS["Fast scharf (0,046)"]))
    assert an.kz.eintraege > an.ref.eintraege_lp and an.kz.luecke < 0.01
    kind, verdikt = E.verdikt_ansicht(an, 0.046)
    assert f"hat der Plan {an.kz.eintraege} Einträge über 1e-6 (die Ecke des LP {an.ref.eintraege_lp})" in verdikt
