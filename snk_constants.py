"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (alles numpy, Zeiten auf dem Entwicklungsrechner, ein Kern; auf der Cloud wahrscheinlich 2- bis 3-mal länger):
- Eine Sinkhorn-Iteration in der Log-Domäne kostet O(n · m) (zwei Log-Sum-Exp), etwa 25 µs bei n = 12 bis 20; bei ε/OPT = 0,01 braucht sie bis zu 30 000 Iterationen.
- Die Reihen (30 Instanzen) laufen bei n = 20 und ε/OPT von 0,014 bis 46 einige Sekunden (Spinner, Zwischenspeicher); der Plan zu kleinem ε/OPT in den
  Experimenten 4 und 5 kommt aus dem Newton-Verfahren auf dem Dual (O((n + m)³) je Schritt, wenige Schritte), nicht aus der Iteration.
- Experiment 2 rechnet bis n = 80 (Transportsimplex etwa 70 ms je Instanz, Newton-Verfahren etwa 30 ms)."""

import math

KOSTENARTEN = ("quadrierter Abstand", "Abstand")

QUELLE_ZUFALL = "Zufallsinstanz"
QUELLE_H1 = "Handbeispiel H1: 2 × 2, gleiche Massen"
QUELLE_H3 = "Handbeispiel H3: 2 × 2, Massen ¾ und ¼"
QUELLEN = (QUELLE_ZUFALL, QUELLE_H1, QUELLE_H3)

EPS_H1 = 1.0 / math.log(3.0)          # ε/OPT des Handbeispiels H1: K = [[1/3, 1/9], [1/9, 1/3]]
EPS_H3 = 2.0 / math.log(3.0)          # ε/OPT des Handbeispiels H3: K = [[1, 1/3], [1/3, 1]]
EPS_REL_STUFEN = tuple(sorted({0.01, 0.014, 0.02, 0.03, 0.046, 0.07, 0.1, 0.14, 0.2, 0.3, 0.46, 0.7, 1.0, 1.4, 2.0, 3.0, 4.6, 7.0, 10.0, 14.0, 20.0, 30.0,
                               46.0, 70.0, 100.0, 140.0, 200.0, EPS_H1, EPS_H3}))
EPS_ROH_STUFEN = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
K_STUFEN = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000)
INSTANZEN2_STUFEN = (10, 20, 30)
INSTANZEN5_STUFEN = (5, 10, 20)
N2_STUFEN = (10, 20, 40, 80)           # Experiment 2: die Reihe läuft über diese n
PFAD_ARTEN = ("Handbeispiel 2 × 2", "20 × 20: C₀ → C₁")
PFAD_T_STUFEN = (50, 200, 800)
VERZERRUNG_KOSTEN = ("stetig (Punkte im Quadrat)", "Gitterkosten (6 × 6)")

N_MIN, N_MAX, DEFAULT_N = 6, 20, 12               # n = m (Lager = Kunden)
SEED_MAX, DEFAULT_SEED = 999999, 35
N_PFAD = 20                                       # Experiment 4, Pfadart 20 × 20
N_VERZERRUNG = 20                                 # Experiment 5
GITTER_SEITE = 6                                  # Experiment 5, Gitterkosten
S_PFAD = 0.5                                      # Massen der Pfade und der Verzerrungsreihe: 0,5/n + 0,5 · Dirichlet(1)
ANZAHL_REIHE = 30                                 # Instanzen der Reihen (Experimente 1 und 3)
SERIEN_SEED = 2026                                # Zufalls-Seed der Reihen: nie der Regler „Zufalls-Seed“
ANZAHL_KONTROLLE = 100                            # Instanzen je n der Kontrollreihe zu Experiment 2 (die Reihe mit 30 Instanzen steigt nicht einheitlich mit n)
KONTROLL_SEED = 777                               # anderer Zufalls-Seed der Kontrollreihe
KONTROLL_MEDIAN = (0.004287717291083699, 0.005984605675543771, 0.0082003085067732, 0.009986874707487851)   # Median der Lücke bei ε/OPT = 0,07 je n (10, 20, 40, 80), quadrierter Abstand: `skala_kontrolle_777` in tools/messreihen.json
EPS_REIHE = (46.0, 4.6, 0.46, 0.046)              # ε/OPT-Stufen der Reihe von Experiment 1
EPS_PRODUKTE = (1.4, 0.46, 0.14, 0.046, 0.014)    # ε/OPT-Stufen der Produktzählung (Experiment 3), je Stufe etwa um den Faktor 3 in ε
EPS_VERZERRUNG = tuple(10 ** (-0.25 * i) for i in range(0, 11))   # ε/OPT 1 bis 0,0032 (Experiment 5)
MAX_ITERATIONEN = 30000
TOL = 1e-8                                        # Abbruch bei Zeilenfehler (L1) unter 1e-8
SCHWELLE_EINTRAG = 1e-6                           # Einträge „über 1e-6“

ZWECK_ANSICHT, ZWECK_REIHE, ZWECK_SKALA, ZWECK_PFAD, ZWECK_VERZERRUNG, ZWECK_PRODUKTE = 1, 2, 3, 4, 5, 6   # getrennte Zufallsströme

COLOR_LAGER = "#2F6B65"
COLOR_KUNDE = "#c0504d"
COLOR_PFEIL = "rgba(47,107,101,0.75)"
COLOR_LINE = "#4c78a8"
COLOR_LP = "#e08a1e"
COLOR_BAND = "rgba(76,120,168,0.18)"

DEFAULTS = {
    "quelle": QUELLE_ZUFALL,
    "kosten_art": KOSTENARTEN[0],
    "n_slider": DEFAULT_N,
    "eps_rel": 0.46,
    "seed_input": DEFAULT_SEED,
    "eps_roh": 0.005,
    "inst2": 30,
    "k_iter": 30,
    "pfad_art": PFAD_ARTEN[1],
    "pfad_T": 200,
    "pfad_t": 0.5,
    "verz_kosten": VERZERRUNG_KOSTEN[0],
    "inst5": 20,
}

# Presets: jede Zeile legt ALLE Regler fest (Schlüssel wie in DEFAULTS); `aktives_preset` vergleicht jeden Regler.
PRESETS = {
    "Von Hand: 2 × 2": dict(DEFAULTS, quelle=QUELLE_H1, eps_rel=EPS_H1),
    "Fast Produktmaß (ε/OPT 46)": dict(DEFAULTS, eps_rel=46.0),
    "Weich (0,46)": dict(DEFAULTS, eps_rel=0.46),
    "Fast scharf (0,046)": dict(DEFAULTS, eps_rel=0.046),
    "Abbruch nach 1 Iteration": dict(DEFAULTS, eps_rel=0.46, k_iter=1),
    "Pfad: Gleichstand": dict(DEFAULTS, pfad_art=PFAD_ARTEN[0], pfad_t=0.5, eps_rel=0.1),
    "Gitterkosten": dict(DEFAULTS, verz_kosten=VERZERRUNG_KOSTEN[1]),
}
PRESET_HELP = {
    "Von Hand: 2 × 2": "Zwei Lager, zwei Kunden, je Hälfte der Masse; ε/OPT = 1/ln 3, sodass die Rechnung in Brüchen aufgeht. Die Handrechnung steht im Expander „Mathematische Formulierung“.",
    "Fast Produktmaß (ε/OPT 46)": "Sehr weicher Plan: kaum ein Unterschied zum Produktmaß a·bᵀ, die Kosten liegen weit über dem LP-Optimum.",
    "Weich (0,46)": "Mittlere Weichheit, die Voreinstellung: der Plan ist dicht, die Kostenlücke noch deutlich.",
    "Fast scharf (0,046)": "Kleines ε/OPT: die Kostenlücke ist klein, der Plan hat aber noch mehr Einträge über 1e-6 als die Ecke des LP (die erste Reihe unten zählt es über viele Instanzen).",
    "Abbruch nach 1 Iteration": "Die Iteration wird nach einem Durchlauf angehalten (zwei Matrix-Vektor-Produkte): die Zeilensummen stimmen noch nicht, die rohen Kosten können unter dem LP-Optimum liegen.",
    "Pfad: Gleichstand": "Experiment 4 am Handbeispiel 2 × 2: beim Pfadwert t = 0,5 ist jede Kopplung optimal, der LP-Plan springt, der weiche Plan ändert sich stetig.",
    "Gitterkosten": "Experiment 5 mit Gitterkosten: es gibt mehrere optimale Pläne, die Kostenlücke fällt dort anders als bei stetigen Kosten.",
}
