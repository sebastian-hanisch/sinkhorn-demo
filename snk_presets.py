"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler hat einen Eintrag in SETTING_SPECS (Name im Session State = Schlüssel in `snk_constants.DEFAULTS`). Reglerwerte aus der Adresszeile werden auf die
Grenzen begrenzt (Zahlen) beziehungsweise auf den nächsten erlaubten Wert eingerastet (Stufen und Auswahllisten); ungültige Werte werden ignoriert."""

import random
from dataclasses import dataclass

import streamlit as st

import snk_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # erlaubte Werte (Auswahllisten und Stufen)
    step: int = 1                     # Schrittweite des Reglers (Werte aus der Adresszeile rasten darauf ein)


SETTING_SPECS = {
    "quelle": SettingSpec("q", str, C.DEFAULTS["quelle"], options=C.QUELLEN),
    "kosten_art": SettingSpec("kosten", str, C.DEFAULTS["kosten_art"], options=C.KOSTENARTEN),
    "n_slider": SettingSpec("n", int, C.DEFAULTS["n_slider"], C.N_MIN, C.N_MAX),
    "eps_rel": SettingSpec("eps", float, C.DEFAULTS["eps_rel"], options=C.EPS_REL_STUFEN),
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed_input"], 0, C.SEED_MAX),
    "eps_roh": SettingSpec("er", float, C.DEFAULTS["eps_roh"], options=C.EPS_ROH_STUFEN),
    "inst2": SettingSpec("i2", int, C.DEFAULTS["inst2"], options=C.INSTANZEN2_STUFEN),
    "k_iter": SettingSpec("k", int, C.DEFAULTS["k_iter"], options=C.K_STUFEN),
    "pfad_art": SettingSpec("pfad", str, C.DEFAULTS["pfad_art"], options=C.PFAD_ARTEN),
    "pfad_T": SettingSpec("T", int, C.DEFAULTS["pfad_T"], options=C.PFAD_T_STUFEN),
    "pfad_t": SettingSpec("t", float, C.DEFAULTS["pfad_t"], 0.0, 1.0),
    "verz_kosten": SettingSpec("vk", str, C.DEFAULTS["verz_kosten"], options=C.VERZERRUNG_KOSTEN),
    "inst5": SettingSpec("i5", int, C.DEFAULTS["inst5"], options=C.INSTANZEN5_STUFEN),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _eingerastet(spec: SettingSpec, wert):
    """Wert auf Grenzen begrenzt beziehungsweise auf die nächste erlaubte Option eingerastet; None, wenn nichts passt."""
    if spec.options is not None:
        if spec.caster is str:
            return wert if wert in spec.options else None
        return min(spec.options, key=lambda o: abs(o - wert))
    if spec.step > 1 and spec.lo is not None:
        wert = spec.lo + round((wert - spec.lo) / spec.step) * spec.step
    if spec.lo is not None:
        wert = max(spec.lo, wert)
    if spec.hi is not None:
        wert = min(spec.hi, wert)
    if spec.caster is float:
        wert = round(wert, 2)       # Regler mit Schrittweite 0,01
    return wert


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (begrenzt beziehungsweise eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                wert = _eingerastet(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if wert is not None:
                st.session_state[state_key] = wert
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
