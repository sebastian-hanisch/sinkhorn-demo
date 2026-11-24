"""Sinkhorn: weich statt scharf - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Der billigste Transportplan ist eine dünne Ecke des Linearen Programms; die Entropie-Regularisierung ersetzt ihn durch einen dichten, weichen Plan, den die
Sinkhorn-Iteration berechnet. Fünf Experimente zählen, was ε bedeutet: Skala, Matrix-Vektor-Produkte, rohe Kostenlücke, Glätte, Verzerrung.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import snk_constants as C
import snk_evaluation as E
from snk_layout import dunkles_design, kopf_css
from snk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from snk_scenario import make_pfad
from snk_visualization import (
    build_eps_reihe,
    build_pfad,
    build_plan_karte,
    build_plan_matrix,
    build_produkte,
    build_roh_verlauf,
    build_skala,
    build_verzerrung,
    build_zeilenfehler,
)

st.set_page_config(page_title="Sinkhorn: weich statt scharf – Sebastian Hanisch", layout="wide")

SPINNER = "Die Reihe wird gerechnet (bei den Standardeinstellungen liegt sie vorgerechnet bereit) ..."


@st.cache_data(show_spinner=False)
def _ansicht(settings):
    return E.ansicht(settings)


@st.cache_data(show_spinner=SPINNER)
def _reihe_eps(n, art):
    return E.eps_v(n, art)


@st.cache_data(show_spinner=SPINNER)
def _skala(eps_roh, anzahl, art):
    return E.skala_v(eps_roh, anzahl, art)


@st.cache_data(show_spinner=SPINNER)
def _roh(n, art, eps_rel):
    return E.roh_v(n, art, eps_rel)


@st.cache_data(show_spinner=SPINNER)
def _produkte(n, art):
    return E.produkte_v(n, art)


@st.cache_data(show_spinner=False)
def _einzel(settings):
    an = E.ansicht(settings)
    return E.rohe_luecke_verlauf(an.inst, an.ref, settings.eps_rel, C.K_STUFEN)


@st.cache_data(show_spinner=SPINNER)
def _pfad_reihe(art, seed, T):
    return E.pfad_v(C.N_PFAD, art, seed, T)


@st.cache_data(show_spinner=False)
def _pfad_2x2(T, eps_rel):
    return E.pfad_2x2(T, eps_rel)[0]


@st.cache_data(show_spinner=False)
def _pfad_plaene(art, seed, t, eps_rel, T, zwei_mal_zwei, eps_bezug):
    if zwei_mal_zwei:
        a, b, C0, C1 = E.pfad_2x2(2, eps_rel)[1]
        eps = eps_rel * 0.5
    else:
        a, b, C0, C1 = make_pfad(C.N_PFAD, art, seed, 0)
        eps = eps_rel * eps_bezug
    return E.pfad_plaene_bei(a, b, C0, C1, t, eps), (a, b)


@st.cache_data(show_spinner=SPINNER)
def _verzerrung(kosten, anzahl):
    return E.reihe_verzerrung(kosten, anzahl)


def md_tabelle(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


def band_text(b, stellen=1):
    """Median [Minimum, Maximum] als Text."""
    return f"{E.de_g(b[0], 3)} [{E.de_g(b[1], 3)}; {E.de_g(b[2], 3)}]"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in snk_layout.py).
st.html(kopf_css(dunkles_design()))
# Breite Tabellen bekommen einen eigenen waagerechten Rollbalken, damit auf schmalen Bildschirmen nicht die ganze Seite quer rollt.
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("🌫️ Sinkhorn: weich statt scharf")
st.markdown(
    """
Ein **Transportplan** (englisch transport plan, auch **Kopplung**, englisch coupling) sagt, wie viel von jedem **Lager** zu jedem **Kunden** fährt, sodass die **Randverteilungen**
(englisch marginals) stimmen: die Zeilensummen sind die Angebote, die Spaltensummen die Nachfragen. Der billigste Plan ist eine dünne Ecke des Linearen Programms (LP). Die
**Entropie-Regularisierung** (englisch entropic regularization) ersetzt ihn durch einen **dichten, weichen Plan**, den die **Sinkhorn-Iteration** berechnet; wie weich er wird, steuert
ε, und die Demo misst ε als Regler immer **relativ zum LP-Optimum (ε/OPT)**; nur Experiment 2 zeigt ein festes rohes ε als Gegenbeispiel. Die fünf Experimente prüfen eine Annahme: *ε ist eine relative Genauigkeit, wie bei einem Toleranzwert.* Absolut gilt zwar 0 ≤ Kosten − OPT ≤ ε · (h(a) + h(b)) mit der Entropie h der Randverteilungen, aber ε hat die Einheit der Kosten, und die relative Lücke hängt am LP-Optimum der Instanz.

Weiter unten: **❓ Kernfragen**, die fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und
**📐 Mathematische Formulierung** (mit der Handrechnung in Brüchen). Gezählt wird in **Matrix-Vektor-Produkten**: eine Iteration sind zwei.
"""
)

with st.expander("So funktioniert die Sinkhorn-Iteration", expanded=True):
    st.markdown(
        """
1. **Aufgabe.** Angebote $a$ (n Lager), Nachfragen $b$ (m Kunden), Kosten $C$ (größter Eintrag 1). Der weiche Plan minimiert $\\langle C, P\\rangle + \\varepsilon\\,\\mathrm{KL}(P \\,\\|\\, a b^\\top)$
   über alle Pläne mit den Randverteilungen $a$ und $b$; $a b^\\top$ ist das **Produktmaß** (englisch product measure).
2. **Gibbs-Form.** Die Lösung hat die Gestalt $P_{ij} = u_i\\, e^{-C_{ij}/\\varepsilon}\\, v_j$: jeder Eintrag ist mathematisch positiv, der Plan ist **dicht** (der Träger ist die ganze Matrix); in der Anzeige erscheinen Einträge unter 1e-6 als leer.
3. **Iteration.** Abwechselnd werden die Zeilen auf $a$ und die Spalten auf $b$ skaliert (**ein Durchlauf = zwei Matrix-Vektor-Produkte**). Nach dem Spaltenschritt stimmen die Spaltensummen genau,
   die Zeilensummen erst am Ende. Die Demo rechnet in der **Log-Domäne** mit den **Kantorovich-Potentialen** $f, g$ (englisch Kantorovich potentials), damit kleines ε keinen Unterlauf erzeugt.
4. **Skala.** ε hat die Einheit der Kosten. Zwei Instanzen mit demselben rohen ε sind verschieden weich; darum der Regler **ε/OPT** mit dem LP-Optimum OPT der Instanz.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧩 Instanz und Weichheit")
    st.selectbox("Instanz oben", C.QUELLEN, key="quelle",
                 help="Die Ansicht oben und die Kennzahlen zeigen ein Handbeispiel oder eine Zufallsinstanz. n, Kostenart und ε/OPT gelten auch für die Reihen der Experimente 1 und 3.")
    st.selectbox("Kostenart", C.KOSTENARTEN, key="kosten_art",
                 help="Kosten = quadrierter Abstand oder Abstand der Orte im Einheitsquadrat, durch den größten Wert geteilt (größter Eintrag 1). Gilt für die Zufallsinstanz, die Reihen und Experiment 4 (20 × 20).")
    st.slider("Lager = Kunden (n)", *bounds("n_slider"), key="n_slider",
              help="Zahl der Lager; ebenso viele Kunden. Wirkt auf die Zufallsinstanz und auf die Reihen der Experimente 1 und 3 (die Handbeispiele haben n = 2).")
    st.select_slider("Weichheit ε/OPT", options=C.EPS_REL_STUFEN, key="eps_rel", format_func=E.eps_text,
                     help="ε geteilt durch das LP-Optimum derselben Instanz. Groß: weicher Plan nahe am Produktmaß. Klein: nahe an der Ecke des LP. Wirkt auf Ansicht, Experiment 3 und Experiment 4.")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.header("🔬 Experiment 2: Skala")
    st.select_slider("festes rohes ε", options=C.EPS_ROH_STUFEN, key="eps_roh", format_func=lambda x: E.de_g(x, 2),
                     help="Ein ε in den Einheiten der Kosten (größter Eintrag 1), das für n = 10, 20, 40 und 80 gleich bleibt.")
    st.select_slider("Instanzen je n", options=C.INSTANZEN2_STUFEN, key="inst2", help="So viele Instanzen je Größe n (Zähleinheit ist die Instanz).")
    st.header("🔬 Experiment 3: Iterationen")
    st.select_slider("Zahl der Iterationen k", options=C.K_STUFEN, key="k_iter",
                     help="Der Plan nach k Durchläufen (2k Produkte), ohne bis zum Ende zu rechnen. Gilt bei der Weichheit ε/OPT und der Instanz der Seitenleiste.")
    st.header("🔬 Experiment 4: Pfad")
    st.radio("Pfadart", C.PFAD_ARTEN, key="pfad_art",
             help="Handbeispiel 2 × 2: C(t) = [[t, 1 − t], [1 − t, t]]. 20 × 20: C(t) = C₀ + t (C₁ − C₀) mit zwei unabhängigen Kostenmatrizen (n = 20 fest, Kostenart der Seitenleiste).")
    st.select_slider("Schrittzahl T", options=C.PFAD_T_STUFEN, key="pfad_T", help="Der Pfad t = 0 bis 1 wird in T Schritten durchlaufen. Bei T = 800 braucht die Reihe 20 × 20 einige Sekunden.")
    st.slider("Pfadwert t", 0.0, 1.0, key="pfad_t", step=0.01, help="Die Pläne zu diesem Wert stehen unter dem Diagramm; die Gleichstand-Stelle des Handbeispiels ist t = 0,5.")
    st.header("🔬 Experiment 5: Verzerrung")
    st.radio("Kosten", C.VERZERRUNG_KOSTEN, key="verz_kosten",
             help="Stetige Kosten (Punkte im Quadrat, eindeutiges Optimum) oder Gitterkosten (L1-Abstand auf einem 6 × 6-Gitter, mehrere optimale Pläne). n = 20 fest.")
    st.select_slider("Instanzen", options=C.INSTANZEN5_STUFEN, key="inst5", help="So viele Instanzen der Reihe (Zähleinheit ist die Instanz).")

sync_query_params({k: st.session_state[k] for k in C.DEFAULTS})

settings = E.settings_aus_zustand(st.session_state)
zwei_mal_zwei = settings.pfad_art == C.PFAD_ARTEN[0]

an = _ansicht(settings)
reihe_eps = _reihe_eps(settings.n, settings.kosten_art)
skala = _skala(settings.eps_roh, settings.inst2, settings.kosten_art)
roh = _roh(settings.n, settings.kosten_art, settings.eps_rel)
einzel = _einzel(settings)
produkte = _produkte(settings.n, settings.kosten_art)
if zwei_mal_zwei:
    pfad = [_pfad_2x2(settings.pfad_T, settings.eps_rel)]
else:
    pfad = _pfad_reihe(settings.kosten_art, settings.seed, settings.pfad_T)
verz = _verzerrung(settings.verz_kosten, settings.inst5)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: der weiche Plan")
kz = an.kz
st.caption(f"{an.inst.name or 'Instanz'}: {an.inst.n} Lager, {an.inst.m} Kunden; OPT = {E.de_g(an.ref.opt, 4)} (LP-Optimum bei größtem Kostenwert 1); "
           f"ε = {E.de_g(an.eps, 3)}, also ε/OPT = {E.eps_text(settings.eps_rel)}.")
m1, m2, m3 = st.columns(3)
m1.metric("Kostenlücke (⟨C, P⟩ − OPT) / OPT", E.prozent(kz.luecke, 2), delta=f"Kosten des Plans = {E.de(kz.kosten / an.ref.opt, 2)} × OPT", delta_color="off")
m2.metric("Abstand zum Produktmaß (Totalvariation)", E.de_g(kz.tv, 3), delta="0 = Produktmaß, 1 = größtmöglich", delta_color="off")
m3.metric("Anteil θ am Weg LP → Produktmaß", E.de_g(kz.theta, 3) if kz.theta is not None else "–", delta="0 = LP-Optimum, 1 = Produktmaß", delta_color="off")
m4, m5, m6 = st.columns(3)
m4.metric("Einträge über 1e-6", f"{kz.eintraege}", delta=f"die Ecke des LP hat {an.ref.eintraege_lp}", delta_color="off")
m5.metric("Matrix-Vektor-Produkte bis Zeilenfehler 1e-8", f"{an.produkte:,}".replace(",", "."), delta=f"{an.iterationen} Iterationen", delta_color="off")
m6.metric("Zeilenfehler (L1)", E.de_g(an.zeilenfehler, 2), delta="Abbruch unter 1e-8", delta_color="off")
if not an.konvergiert:
    st.warning(f"Nicht konvergiert: nach {C.MAX_ITERATIONEN:,} Iterationen".replace(",", ".") + f" liegt der Zeilenfehler bei {E.de_g(an.zeilenfehler, 2)}; gezeigt ist der Plan zu diesem Stand "
               "(bei sehr kleinem ε/OPT braucht die Iteration sehr viele Durchläufe).")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_plan_matrix(an.plan, an.inst.a, an.inst.b, titel=f"weicher Plan, ε/OPT = {E.eps_text(settings.eps_rel)} (Zahl = Anteil der Gesamtmasse in %)"),
                    width="stretch", key="plan_weich")
with c2:
    st.plotly_chart(build_plan_matrix(an.ref.plan_lp, an.inst.a, an.inst.b, titel="LP-Plan (Ecke)"), width="stretch", key="plan_lp")
if an.inst.lager is not None:
    k1, k2 = st.columns(2)
    with k1:
        st.plotly_chart(build_plan_karte(an.inst.lager, an.inst.kunden, an.plan, titel="weicher Plan als Pfeile"), width="stretch", key="karte_weich")
    with k2:
        st.plotly_chart(build_plan_karte(an.inst.lager, an.inst.kunden, an.ref.plan_lp, titel="LP-Plan als Pfeile", farbe="rgba(224,138,30,0.85)"), width="stretch", key="karte_lp")
st.caption("Zeilen = Lager (Angebot in % in der Beschriftung), Spalten = Kunden (Nachfrage in %). Leere Zellen: Einträge unter 1e-6. Pfeile: Einträge über 0,1 % der Gesamtmasse, Breite wächst mit dem Eintrag.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**Wie weich ist der Plan bei diesem ε/OPT?**")
kind, text = E.verdikt_ansicht(an, settings.eps_rel)
getattr(st, kind)(text)
st.markdown("**Wie viele Produkte kostet er?**")
kind, text = E.verdikt_ansicht_produkte(an, settings.eps_rel)
getattr(st, kind)(text)
st.markdown("**Springt er, wenn sich die Kosten ändern?**")
kind, text = E.verdikt_pfad(pfad, settings.pfad_T, zwei_mal_zwei)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Vom scharfen zum weichen Plan")
st.caption(f"Reihe über {reihe_eps[0]['anzahl']} feste Instanzen (n = {settings.n}, {settings.kosten_art}, Zufalls-Seed der Reihen {C.SERIEN_SEED}); Median [Minimum; Maximum]; "
           f"Abbruch bei Zeilenfehler unter 1e-8, Produkte = 2 × Iterationen (ohne das eine Produkt des Abbruchtests). Der Regler ε/OPT oben wirkt auf die Ansicht, die Reihe läuft über feste Stufen.")
st.markdown(md_tabelle(
    ["ε/OPT", "Abstand zum Produktmaß (TV)", "Kosten des Plans / OPT", "Kostenlücke", "Einträge über 1e-6 (Ecke)", "Produkte"],
    [[E.eps_text(z["eps_rel"]), band_text(z["tv"]), band_text(z["fach"]), E.prozent(z["luecke"][0], 2), f"{E.band_ganz(z['eintraege'])} ({E.ganz(z['eintraege_lp'][0])})", E.band_ganz(z["produkte"])]
     for z in reihe_eps]))
st.plotly_chart(build_eps_reihe(reihe_eps), width="stretch", key="eps_reihe")
kind, text = E.verdikt_vom_scharfen_zum_weichen(reihe_eps)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Skala – ε/OPT, nicht ε")
st.caption(f"Je n = {', '.join(str(z['n']) for z in skala)} {skala[0]['anzahl']} Instanzen ({settings.kosten_art}, Zufalls-Seed der Reihen {C.SERIEN_SEED}); ε/OPT = 0,07 für die rechte Abbildung wird mit dem Newton-Verfahren "
           "auf dem Dual exakt gelöst (kein Iterationslimit).")
s1, s2 = build_skala(skala, settings.eps_roh, E.EPS_REL_FEST)
sc1, sc2 = st.columns(2)
with sc1:
    st.plotly_chart(s1, width="stretch", key="skala_eps")
with sc2:
    st.plotly_chart(s2, width="stretch", key="skala_luecke")
st.markdown(md_tabelle(
    ["n", "LP-Optimum OPT", f"ε = {E.de_g(settings.eps_roh, 2)} als ε/OPT", "Lücke bei ε/OPT = 0,07"],
    [[str(z["n"]), band_text(z["opt"]), band_text(z["eps_rel_fest"]), (band_text([100 * w for w in z["luecke"]]) + " %") if z["luecke"] else "–"] for z in skala]))
kind, text = E.verdikt_skala(skala, settings.eps_roh)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie viele Produkte, und was die Kosten vor dem Ende zeigen")
st.caption(f"n = {settings.n}, {roh['anzahl']} Instanzen, ε/OPT = {E.eps_text(settings.eps_rel)}: Plan nach k Iterationen (nach dem Spaltenschritt; k = 2 heißt vier Produkte). "
           "Strichlinie: k der Seitenleiste; gepunktet: die Instanz der Seitenleiste.")
rc1, rc2 = st.columns(2)
with rc1:
    st.plotly_chart(build_roh_verlauf(roh, einzel, settings.k_iter), width="stretch", key="roh_verlauf")
with rc2:
    st.plotly_chart(build_zeilenfehler(roh, settings.k_iter), width="stretch", key="zeilenfehler")
i = roh["k"].index(settings.k_iter)
st.markdown(f"**Nach k = {settings.k_iter} Iterationen ({2 * settings.k_iter} Produkte):** rohe Kostenlücke im Median {E.prozent(roh['luecke'][i][0], 1)} "
            f"(Spanne {E.prozent(roh['luecke'][i][1], 1)} bis {E.prozent(roh['luecke'][i][2], 1)}), Kosten unter OPT in {roh['negativ'][i]} von {roh['anzahl']} Instanzen; "
            f"Zeilenfehler im Median {E.de_g(roh['zeilenfehler'][i][0], 2)}. Die rohen Kosten sind dann keine Kosten einer Kopplung: die Zeilensummen stimmen nicht.")
pz = produkte["zeilen"]
st.plotly_chart(build_produkte(produkte), width="stretch", key="produkte")
st.markdown(md_tabelle(
    ["ε/OPT", "Produkte bis Zeilenfehler 1e-8", "nicht konvergiert", "Faktor zur Stufe davor (Mediane)"],
    [[E.eps_text(z["eps_rel"]), E.band_ganz(z["produkte"]), f"{z['nicht_konvergiert']} von {produkte['anzahl']}",
      (E.de(produkte["faktoren"][j - 1], 2) + f" (ε-Faktor {E.de(produkte['eps_faktoren'][j - 1], 2)})") if j > 0 else "–"] for j, z in enumerate(pz)]))
kind, text = E.verdikt_produkte(roh, produkte, settings.k_iter)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Glatt statt springend")
if zwei_mal_zwei:
    st.caption(f"Handbeispiel 2 × 2, C(t) = [[t, 1 − t], [1 − t, t]], a = b = (½, ½), ε = ε/OPT · ½ (½ ist das Optimum bei t = ½), T = {settings.pfad_T} Schritte.")
else:
    st.caption(f"n = {C.N_PFAD}, {len(pfad)} Pfade C(t) = C₀ + t (C₁ − C₀) aus zwei unabhängigen Kostenmatrizen ({settings.kosten_art}), Massen 0,5/n + 0,5 · Dirichlet(1), T = {settings.pfad_T} Schritte, "
               f"Zufalls-Seed {settings.seed}; ε = ε/OPT · mittleres OPT entlang des Pfads. Das Diagramm zeigt den ersten Pfad.")
st.plotly_chart(build_pfad(pfad[0], zwei_mal_zwei, settings.pfad_t), width="stretch", key="pfad")
eps_bezug = pfad[0]["eps_bezug"]
(lp_t, weich_t), (pa, pb) = _pfad_plaene(settings.kosten_art, settings.seed, settings.pfad_t, settings.eps_rel, settings.pfad_T, zwei_mal_zwei, eps_bezug)
pc1, pc2 = st.columns(2)
with pc1:
    st.plotly_chart(build_plan_matrix(lp_t, pa, pb, titel=f"LP-Plan bei t = {E.de(settings.pfad_t, 2)}"), width="stretch", key="pfad_lp")
with pc2:
    if weich_t is not None:
        st.plotly_chart(build_plan_matrix(weich_t, pa, pb, titel=f"weicher Plan bei t = {E.de(settings.pfad_t, 2)}, ε/OPT = {E.eps_text(settings.eps_rel)}"), width="stretch", key="pfad_weich")
    else:
        st.warning("Das Newton-Verfahren ist bei diesem Pfadwert nicht konvergiert.")
if zwei_mal_zwei:
    w = pfad[0]["weich"][0]
    if w.get("ok"):
        st.markdown(f"**Sprünge:** LP-Plan {E.de(pfad[0]['lp_max_sprung'], 2)} (ein Schritt), weicher Plan {E.de(w['max_sprung'], 3)}; ε·L = {E.de(w['eps_L'], 3)} mit L = größter L1-Sprung / größte Kostenänderung je Schritt.")
else:
    zeilen = []
    for j, er in enumerate(E.PFAD_EPS_REL):
        ws = [x for e in pfad for x in e["weich"] if abs(x["eps_rel"] - er) < 1e-12 and x.get("ok")]
        if ws:
            zeilen.append([E.eps_text(er), band_text(E.band([x["max_sprung"] for x in ws])), band_text(E.band([x["eps_L"] for x in ws])), f"{len(ws)} von {len(pfad)}"])
    st.markdown("**Weicher Plan, größter L1-Sprung je Schritt** (Median [Minimum; Maximum] über die Pfade; ε·L mit L = größter L1-Sprung / größte Kostenänderung je Schritt):")
    st.markdown(md_tabelle(["ε/OPT", "größter Sprung", "ε·L", "Pfade konvergiert"], zeilen))
    st.markdown(f"**LP-Plan:** Trägerwechsel (Schritte, in denen sich die Menge der Einträge über 1e-9 ändert; eine Untergrenze der Basiswechsel) {band_text(E.band([e['lp_wechsel'] for e in pfad]))}, "
                f"größter L1-Sprung {band_text(E.band([e['lp_max_sprung'] for e in pfad]))} (Höchstwert 2).")
kind, text = E.verdikt_pfad(pfad, settings.pfad_T, zwei_mal_zwei)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Wie schnell verschwindet die Verzerrung?")
st.caption(f"n = {C.N_VERZERRUNG}, {verz['anzahl']} Instanzen ({settings.verz_kosten}), Massen {'1/n' if verz['kosten'] == C.VERZERRUNG_KOSTEN[1] else '0,5/n + 0,5 · Dirichlet(1)'}; der Plan kommt aus dem Newton-Verfahren auf dem Dual "
           f"mit Fortsetzung in ε (Randfehler unter 1e-12). Achsen logarithmisch; Lücken unter 1e-11 sind vom Rechenrauschen nicht zu trennen und fehlen. Mitte der Fläche und Konvergenz bei Gleichstand: siehe „Wo die Annahmen enden“.")
st.plotly_chart(build_verzerrung(verz), width="stretch", key="verzerrung")
sl = verz["steigung"]
st.markdown(md_tabelle(
    ["ε/OPT", "Median der Kostenlücke", "Spanne", "Steigung d ln Lücke / d ln ε bis zur nächsten Stufe"],
    [[E.eps_text(e), E.de_g(m, 3), f"{E.de_g(lo, 2)} bis {E.de_g(hi, 2)}", (E.de(s, 2) if s is not None else "–")]
     for e, m, lo, hi, s in zip(verz["stufen"], verz["median"], verz["minimum"], verz["maximum"], sl + [None]) if m == m and m > E.LUECKE_BODEN]))
st.caption(f"{verz['nicht_konvergiert']} (Instanz, ε)-Paare von {verz['anzahl'] * len(verz['stufen'])} sind nicht konvergiert und fehlen in den Medianen.")
kind, text = E.verdikt_verzerrung(verz)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme dieser Demo | Was passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| **Der optimale Plan ist eine dünne Ecke des LP (höchstens n + m − 1 Einträge), und bei Gleichstand wählt der Löser eine.** | Der weiche Plan ist dicht und eindeutig (Gibbs-Form). Bei Gleichstand ist er im 2 × 2-Beispiel das Produktmaß (Experiment 4); die Wahl bei Gleichstand und die Konvergenz dort werden hier nicht untersucht, nur die Lückenkurve (Experiment 5). | Rechengenauigkeit des Sinkhorn-Verfahrens |
| **Der Kern K = exp(−C/ε) ist überall positiv, beide Randverteilungen haben dieselbe Gesamtmasse, ε ist ein frei gewählter Regler, und „gelaufen“ heißt „gelöst“.** | Nach einer Iteration sind die Zeilensummen falsch, die rohen Kosten können unter OPT liegen (Experiment 3). Nullen im Kern, verschiedene Gesamtmassen und ein aus Beobachtungen zurückgerechnetes ε sind hier nicht untersucht. | Matrix abgleichen (Furness, IPF) |
| **Der gezeigte Plan ist exakt gerechnet: kein Unterlauf, kein Gleichstand, und seine Ableitung ist die des Grenzplans.** | Die Demo rechnet in float64 in der Log-Domäne; endliche Genauigkeit, Unterlauf und die Ableitung am unfertigen Plan sind hier nicht untersucht. | Rechengenauigkeit des Sinkhorn-Verfahrens |
| **⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.** | Die Näherung wird besser (Experiment 5), aber auch dann ist sie kein Abstand: ⟨C, P_ε⟩ ist für gleiche Verteilungen nicht 0. Welche Definition eines „Sinkhorn-Abstands“ gewählt wird, ändert den Wert. Diese Demo benutzt das Wort nicht. | Abstände zwischen Verteilungen |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Lager und Kunden sind Zufallspunkte im Einheitsquadrat, die Kosten der quadrierte Abstand oder der Abstand (durch den größten Wert geteilt, größter Eintrag 1); die Massen sind Dirichlet(1)-verteilt
  (Experimente 4 und 5: 0,5/n + 0,5 · Dirichlet(1)); Gitterkosten nur in Experiment 5. Beide Gesamtmassen sind 1.
- Das LP-Optimum OPT kommt aus einem eigenen Transportsimplex; ε/OPT ist je Instanz mit ihrem eigenen OPT gebildet.
- Zufall aus dem Hausgenerator SplitMix64; die Reihen benutzen feste Instanzen (Zufalls-Seed 2026) unabhängig vom Regler „Zufalls-Seed“.

**Grenzen.**
- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Instanz; Matrix-Vektor-Produkte). Ein Median über 10 bis 30 Instanzen ist kein Gesetz.
- Der Plan zu sehr kleinem ε/OPT (Experimente 4 und 5) kommt aus dem Newton-Verfahren auf dem Dual, nicht aus der Iteration; die Iteration bräuchte dort sehr viele Durchläufe. Gezählt werden Produkte nur für die Iteration.
- „Etwa proportional zu 1/ε“, „Potenz mit lokaler Steigung um 2“ und „exponentiell“ sind Beschreibungen des gemessenen Bereichs (ε/OPT 0,003 bis 1, n = 20), keine Gesetze; die Schwelle, ab der der Verlauf bei eindeutigem Optimum exponentiell wird, ist hier nicht bestimmt.
- Der Anstieg der Lücke mit n bei festem ε/OPT ist bei 30 Instanzen je n in Experiment 2 nicht monoton (Stichprobeneffekt: mit 100 Instanzen je n steigt der Median); über n = 80 hinaus ist er nicht untersucht.
- Verteilungen mit Atomen (Zonenmatrizen), nicht quadrierte Abstände bei Glätte und Steigung, n ≠ m und Gesamtmassen ungleich 1 sind nicht untersucht.

**Bekannte Probleme und Unsicherheiten.**
- Die Zahlen sind mit Zufalls-Seed 2026 gemessen; andere Zufalls-Seeds geben andere Zahlen im Band, nicht dieselben.
- Bei sehr kleinem ε/OPT kann die Iteration das Limit von 30 000 Iterationen erreichen; die App meldet das als „nicht konvergiert“ und zeigt den Plan zu diesem Stand.
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Angebote $a \in \mathbb{R}^n_{>0}$ (Summe 1), Nachfragen $b \in \mathbb{R}^m_{>0}$ (Summe 1), Kosten $C \in \mathbb{R}^{n\times m}$ (größter Eintrag 1). *Gesucht:* ein
Transportplan $P \ge 0$ mit Zeilensummen $a$ und Spaltensummen $b$, scharf mit kleinsten Kosten, weich mit kleinsten Kosten plus Strafe. Entwurfsentscheidungen (Zählung, Skala) stehen unter der Tabelle.

| Symbol | Bedeutung |
|---|---|
| $n$, $m$ | Zahl der Lager und der Kunden |
| $a$, $b$ | Angebote und Nachfragen (Randverteilungen), Summe 1 |
| $C$ | Kosten, $C_{ij}$ je Lager $i$ und Kunde $j$, größter Eintrag 1 |
| $P$ | Transportplan (Kopplung), $P_{ij}$ = Masse von Lager $i$ zu Kunde $j$ |
| $\mathrm{OPT}$ | Optimalwert des LP, $\min \langle C, P\rangle$ |
| $\varepsilon$ | Gewicht der Entropie-Strafe (Einheit der Kosten); $\varepsilon/\mathrm{OPT}$ ist der Regler |
| $K$ | Kern $K_{ij} = e^{-C_{ij}/\varepsilon}$ |
| $u$, $v$ | Skalierungsvektoren der Gibbs-Form $P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)$ |
| $f$, $g$ | Kantorovich-Potentiale der Log-Domäne, $u_i = a_i e^{f_i/\varepsilon}$, $v_j = b_j e^{g_j/\varepsilon}$ |
| $\theta$ | Anteil am Weg vom LP-Optimum zum Produktmaß |

**Problem.** Scharf (LP) und weich:
$$\mathrm{OPT} = \min_{P \ge 0,\; P\mathbf 1 = a,\; P^{\top}\mathbf 1 = b} \langle C, P\rangle$$
$$P_\varepsilon = \operatorname*{arg\,min}_{P \ge 0,\; P\mathbf 1 = a,\; P^{\top}\mathbf 1 = b} \langle C, P\rangle + \varepsilon\,\mathrm{KL}(P \,\|\, a b^{\top}), \qquad \mathrm{KL}(P \,\|\, a b^{\top}) = \sum_{ij} P_{ij} \ln \frac{P_{ij}}{a_i b_j}$$
Gleiche Minimierer wie $\langle C, P\rangle - \varepsilon H(P)$ mit der Entropie $H(P) = \sum_{ij} P_{ij}\ln(1/P_{ij})$ (die Terme von $a$ und $b$ sind Konstanten).

**Gibbs-Form und Iteration.** Die Lösung ist $P_{ij} = u_i K_{ij} v_j$ (Cuturi 2013, Gl. (3)). Die Sinkhorn-Iteration setzt abwechselnd
$$u \leftarrow a \oslash (K v), \qquad v \leftarrow b \oslash (K^{\top} u).$$
In der Log-Domäne mit $P_{ij} = a_i b_j \exp\big((f_i + g_j - C_{ij})/\varepsilon\big)$:
$$f_i \leftarrow -\varepsilon \ln \sum_j b_j\, e^{(g_j - C_{ij})/\varepsilon}, \qquad g_j \leftarrow -\varepsilon \ln \sum_i a_i\, e^{(f_i - C_{ij})/\varepsilon}.$$
**Zählung.** Eine Iteration (ein $f$- und ein $g$-Schritt) sind **zwei Matrix-Vektor-Produkte**; in der Log-Domäne ist jedes ein Log-Sum-Exp über $n \cdot m$ Einträge. Abbruch bei Zeilenfehler
$\|P\mathbf 1 - a\|_1 < 10^{-8}$, gebildet aus dem nächsten $f$-Schritt, den die folgende Iteration wiederverwendet; beim Abbruch bleibt er übrig, ausgewertet werden also $2k+1$ Log-Sum-Exp bei gemeldeten $2k$ Produkten (ein Produkt mehr, als die Zählung nennt). Nach dem $g$-Schritt stimmen die Spaltensummen genau.

**Kennzahlen.** Kostenlücke $(\langle C, P_\varepsilon\rangle - \mathrm{OPT})/\mathrm{OPT}$; Abstand zum Produktmaß $\mathrm{TV} = \tfrac12 \|P - a b^{\top}\|_1$; Anteil
$$\theta = \frac{\langle C, P\rangle - \mathrm{OPT}}{\langle C, a b^{\top}\rangle - \mathrm{OPT}}.$$

**Handbeispiel H1** (zwei Lager, zwei Kunden, $a = b = (\tfrac12, \tfrac12)$, $C = \begin{pmatrix} 1/2 & 1 \\ 1 & 1/2 \end{pmatrix}$): die Pläne sind $P = \begin{pmatrix} x & 1/2 - x \\ 1/2 - x & x \end{pmatrix}$ mit Kosten $1 - x$;
$\mathrm{OPT} = \tfrac12$ bei $x = \tfrac12$, das Produktmaß ($x = \tfrac14$) kostet $\tfrac34$. Mit $\varepsilon = 1/(2\ln 3)$ ist $K = \begin{pmatrix} 1/3 & 1/9 \\ 1/9 & 1/3 \end{pmatrix}$. Ein Durchlauf gibt $u = \tfrac12 / (\tfrac13 + \tfrac19) = \tfrac98$,
danach $v = 1$, also
$$P = \tfrac98 K = \begin{pmatrix} 3/8 & 1/8 \\ 1/8 & 3/8 \end{pmatrix}.$$
Kosten $\tfrac58$, Kostenlücke $\tfrac14 = 25\,\%$, $\theta = \tfrac12$, $\mathrm{TV} = \tfrac14$, $\varepsilon/\mathrm{OPT} = 1/\ln 3 \approx 0{,}91$. Allgemein gilt aus $x/(\tfrac12 - x) = e^{1/(2\varepsilon)}$ die Lücke
$$\text{Lücke}(\varepsilon/\mathrm{OPT} = e) = \frac{1}{1 + e^{1/e}}.$$

**Handbeispiel H3** ($a = (\tfrac34, \tfrac14)$, $b = (\tfrac14, \tfrac34)$, $C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $\mathrm{OPT} = \tfrac12$, $\varepsilon = 1/\ln 3$, also $\varepsilon/\mathrm{OPT} = 2/\ln 3 \approx 1{,}82$): $K = \begin{pmatrix} 1 & 1/3 \\ 1/3 & 1 \end{pmatrix}$.
Nach einem Durchlauf ist $u = (\tfrac{9}{16}, \tfrac{3}{16})$, $K^{\top}u = (\tfrac58, \tfrac38)$, $v = (\tfrac25, 2)$ und
$$P = \begin{pmatrix} 9/40 & 3/8 \\ 1/40 & 3/8 \end{pmatrix}.$$
Die Spaltensummen $(\tfrac14, \tfrac34)$ stimmen, die Zeilensummen sind $(\tfrac35, \tfrac25)$ statt $(\tfrac34, \tfrac14)$ (Zeilenfehler $\tfrac{3}{10}$). Die Kosten $\tfrac25$ liegen unter $\mathrm{OPT} = \tfrac12$: rohe Lücke $-20\,\%$.
Der Grenzplan hat $x = (9 - 3\sqrt 3)/16$ und die Lücke $(3\sqrt 3 - 5)/4 \approx +4{,}9\,\%$. Dasselbe $\varepsilon = 1/(2\ln 3)$ wie bei H1 gibt bei H3 nur die Lücke $(\sqrt{1701} - 41)/40 \approx 0{,}6\,\%$ statt $25\,\%$:
$\varepsilon/\mathrm{OPT}$ allein legt die Lücke nicht fest.

**Handbeispiel H4** (Pfad $C(t) = \begin{pmatrix} t & 1-t \\ 1-t & t \end{pmatrix}$, $a = b = (\tfrac12, \tfrac12)$): der LP-Plan ist die Einheitsmatrix für $t < \tfrac12$ und die Gegendiagonale für $t > \tfrac12$ (L1-Sprung 2). Der weiche Plan hat
$$P_{11}(t) = \tfrac12\,\sigma\!\Big(\frac{1 - 2t}{\varepsilon}\Big), \qquad \sigma(s) = \frac{1}{1 + e^{-s}};$$
bei $t = \tfrac12$ ist er für jedes $\varepsilon$ das Produktmaß (alle Einträge $\tfrac14$). Die Steigung ist dort $-1/(4\varepsilon)$, mit $L = \|\Delta P\|_1/\|\Delta C\|_{\max}$ also $\varepsilon L = 1$.

**KL-Projektion.** Der weiche Plan ist die KL-Projektion von $K$ auf die Pläne mit den Randverteilungen $a$ und $b$ (Benamou u. a. 2015, Gl. (6)); bei H1 liegt das Minimum bei $x = \tfrac38$.

**Schrödinger-Brücke (Deutung, keine Zählung).** Das statische Problem der Entropie-Regularisierung ist dasselbe Optimierungsproblem wie die statische Schrödinger-Brücke zum Referenzmaß mit Dichte $e^{-c}/Z$ (Nutz, Abschn. 4, Gleichungen (4.2) bis (4.4); Schrödinger-Potentiale sind die Sinkhorn-Potentiale). Lesart: ε ist die Temperatur; der weiche Plan ist die wahrscheinlichste Kopplung einer Zufallsbewegung, die an beiden Rändern festgenagelt ist.
Léonard (2014, Abschn. 5): das quadratische Monge-Kantorovich-Problem ist ein Γ-Limes von Schrödinger-Problemen.

**Zur Skala.** ε hat die Einheit der Kosten: Kosten und ε mit demselben Faktor multipliziert geben denselben Plan (bei H1 mit Faktor 10 beide: derselbe Plan; Kosten mal 10 bei gleichem ε: Lücke $1/59\,050$). Darum die Skala ε/OPT; sie ist keine instanzunabhängige Größe (H1 und H3).

**Literatur** (Angaben zu den geprüften Stellen jeweils in Klammern): Cuturi (2013), *Sinkhorn distances: lightspeed computation of optimal transportation distances*, arXiv 1306.0895 (Gl. (2), (3), arXiv-Fassung); Altschuler, Weed, Rigollet (2017), *Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration*, arXiv 1705.09634 (Lemma 1: Skalierungsform); Benamou, Carlier, Cuturi, Nenna, Peyré (2015), *Iterative Bregman projections for regularized transportation problems*, arXiv 1412.5154 (Gl. (6)); Nutz, *Introduction to Entropic Optimal Transport* (Vorlesungsskript 2022, Abschn. 4; Gleichungsnummern hier nicht nachgeprüft); Léonard (2014), *A survey of the Schrödinger problem and some of its connections with optimal transport*, arXiv 1308.0215 (Abschn. 5 laut Gliederung; Wortlaut nicht abschließend geprüft); Peyré, Cuturi (2019), *Computational Optimal Transport* (Hintergrund, Kapitel 1 bis 4; keine Zahl der Demo hängt an ihm). Dieselbe Rechnung heißt in der Verkehrsplanung Furness-Verfahren.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
