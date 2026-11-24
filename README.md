# Sinkhorn: weich statt scharf – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-sinkhorn-demo.streamlit.app/)** -->

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an der Annahme an, die die Wurzel der Reihe („Vom Zuordnen zum Transportieren“) stehen lässt: *Der optimale Plan ist eine dünne Ecke des LP (höchstens n + m − 1 Einträge), und bei Gleichstand wählt der Löser eine.* Hier wird der Plan **weich**: dicht, eindeutig und glatt von den Kosten abhängig. Vehikel ist wieder der Ort: **Lager** mit Angeboten, **Kunden** mit Nachfragen, Kosten je Einheit. Gerechnet wird mit der **Sinkhorn-Iteration** in der Log-Domäne (`snk_sinkhorn.py`), dem Newton-Verfahren auf dem Dual für sehr kleines ε (`snk_newton.py`) und dem eigenen Transportsimplex für das LP-Optimum (`snk_simplex.py`, eine Kopie aus der Wurzel-Demo); alles nur numpy.

## Warum dieses Problem

Ein **Transportplan** (englisch transport plan, auch **Kopplung**, englisch coupling; nicht zu verwechseln mit der Kopplung zweier Markov-Ketten) sagt, wie viel von jedem Lager zu jedem Kunden fährt, sodass die **Randverteilungen** (englisch marginals) stimmen. Die **Entropie-Regularisierung** (englisch entropic regularization) mischt zu den Kosten eine Strafe mit Gewicht ε, die den Plan dem **Produktmaß** (englisch product measure) $a b^{\top}$ annähert, und die Sinkhorn-Iteration löst das weiche Problem. Die naheliegende Lesart, ε sei „eine Genauigkeit wie ein Toleranzwert“, stimmt für die relative Kostenlücke nicht: absolut gilt zwar $0 \le \langle C, P_\varepsilon\rangle - \mathrm{OPT} \le \varepsilon\,(h(a) + h(b))$ mit der Entropie $h$ der Randverteilungen (Test `test_kostenluecke_liegt_zwischen_null_und_eps_mal_der_summe_der_randentropien`), aber **ε hat die Einheit der Kosten, und die relative Lücke hängt am LP-Optimum der Instanz; darum misst die Demo ε relativ zu OPT**. Dasselbe ε/OPT gibt auf zwei Instanzen verschiedene Lücken, ein kleines ε macht den Plan nicht so dünn wie die Ecke und die Iteration nicht billig, und bricht man die Iteration vor dem Ende ab, können die Kosten unter dem LP-Optimum liegen. Die fünf Experimente zählen das.

## Modell

Gegeben sind Angebote $a$ ($n$ Lager, Summe 1), Nachfragen $b$ ($m$ Kunden, Summe 1) und Kosten $C$ ($n \times m$, größter Eintrag 1). Der weiche Plan minimiert $\langle C, P\rangle + \varepsilon\,\mathrm{KL}(P \,\|\, a b^{\top})$ über alle Pläne $P \ge 0$ mit Zeilensummen $a$ und Spaltensummen $b$ und hat die Gibbs-Form $P_{ij} = u_i e^{-C_{ij}/\varepsilon} v_j$ (mathematisch ist jeder Eintrag positiv, der Träger die ganze Matrix; in float64 unterlaufen kleine Einträge zu 0, und die App lässt Einträge unter 1e-6 leer). Die Formeln, die Handrechnung in Brüchen und die Lesehilfe stehen im 📐-Expander der App.

- **Instanzen (Objekt Ort):** $n$ Lager und $n$ Kunden gleichverteilt im Einheitsquadrat, Massen Dirichlet(1) (unabhängige Exponentialzahlen, durch ihre Summe geteilt), Kosten = quadrierter Abstand oder Abstand, durch den größten Wert geteilt (größter Eintrag 1). Experimente 4 und 5 (stetig): Massen $0{,}5/n + 0{,}5 \cdot \text{Dirichlet}(1)$. Gitterkosten (Experiment 5): $n = 20$ Lager und 20 Kunden auf je 20 verschiedenen Zellen eines 6 × 6-Gitters, Kosten = L1-Abstand, Massen $1/n$.
- **Skala ε/OPT:** OPT ist das LP-Optimum **derselben Instanz** bei größtem Kostenwert 1; ε hat die Einheit der Kosten. Der Regler heißt ε/OPT (29 Stufen von 0,01 bis 200), das rohe ε steht daneben. Zahlen zur Weichheit stehen mit ihrer Skala ε/OPT.
- **Zählweise:** Eine Iteration (ein $f$- und ein $g$-Schritt) sind **zwei Matrix-Vektor-Produkte**; in der Log-Domäne ist jedes Produkt ein Log-Sum-Exp über $n \cdot m$ Einträge. In Verdikten und Tabellen stehen **Produkte** (zwei je Iteration, ohne das eine Produkt des Abbruchtests), nicht Iterationen. Abbruch bei Zeilenfehler (L1) unter 1e-8, höchstens 30 000 Iterationen (danach „nicht konvergiert“). Start $u = v = 1$ (Lehrbuchform).
- **Zähleinheit** jeder Messung ist die **Instanz** (Median [Minimum; Maximum] über 30 Instanzen, wo nichts anderes steht); die Reihen benutzen feste Instanzen mit dem Zufalls-Seed 2026, unabhängig vom Regler „Zufalls-Seed“.

## Methodik

- **Sinkhorn-Iteration** (`snk_sinkhorn.py`): Log-Domäne mit den **Kantorovich-Potentialen** $f, g$ (englisch Kantorovich potentials), $P_{ij} = a_i b_j \exp((f_i + g_j - C_{ij})/\varepsilon)$; der Abbruchtest benutzt den nächsten $f$-Schritt, den die folgende Iteration wiederverwendet; beim Abbruch bleibt dieser eine Schritt übrig, ausgewertet werden also 2k + 1 Log-Sum-Exp bei gemeldeten 2k Produkten (ein Produkt mehr, als die Zählung nennt; `test_abbruchtest_kostet_ein_produkt_mehr_als_gemeldet`). Die Log-Domäne löst den Unterlauf von $e^{-C/\varepsilon}$ bei kleinem ε; sie macht die Iteration nicht schneller.
- **Newton-Verfahren auf dem Dual** (`snk_newton.py`): derselbe Plan, anderes Verfahren; mit Fortsetzung in ε (ε absteigend, Start aus der Lösung davor). Experiment 2 (Plan bei ε/OPT = 0,07), 4 und 5 nehmen es, weil die Iteration bei sehr kleinem ε/OPT tausende Durchläufe bräuchte; Produkte werden dort nicht gezählt.
- **LP-Optimum** (`snk_simplex.py`, `snk_eindeutigkeit.py`): der Transportsimplex mit Basisbaum und der exakte Eindeutigkeitstest der Wurzel-Demo, als Kopie mit eigenem Präfix (ein Test vergleicht die Kopie mit der Quelle).
- **Zufall:** Hausgenerator SplitMix64 (`snk_rng.py`, auf Windows und Linux bitgleich).
- **Vorrechnung:** Die Reihen für die Standardeinstellungen liegen in `tools/vorgerechnet.json` (erzeugt von `tools/vorrechnen.py`), damit die Seite sofort erscheint; jede andere Einstellung rechnet live. `tools/messreihen.py` rechnet die Reihen dieser README (auch $n = 40$) und schreibt `tools/messreihen.json`; `tests/test_claims.py` bildet die README-Zahlen aus dieser Datei und rechnet die Einträge der Vorrechnung neu.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `tools/messreihen.json` (Zufalls-Seed 2026); sie gelten **in diesem Modell, bei diesen Parametern** (Lager und Kunden im Einheitsquadrat, Dirichlet(1)-Massen, $n$ wie angegeben, 30 Instanzen) und sind keine Gesetze.

### Experiment 1: Vom scharfen zum weichen Plan

Quadrierter Abstand, Zeilenfehler 1e-8; Spalten: Abstand zum Produktmaß (Totalvariation), Kosten des Plans in Vielfachen von OPT, Kostenlücke (Median), Einträge über 1e-6 (in Klammern die Ecke des LP), Produkte bis Zeilenfehler 1e-8.

**n = 12** (30 Instanzen):

| ε/OPT | Abstand zum Produktmaß (TV) | Kosten des Plans / OPT | Kostenlücke | Einträge über 1e-6 (Ecke) | Produkte |
|---|---|---|---|---|---|
| 46 | 0,0135 [0,00386; 0,0291] | 2,51 [1,48; 4,58] | 150,79 % | 144 [140; 144] (23) | 6 [6; 8] |
| 4,6 | 0,133 [0,0384; 0,24] | 2,05 [1,44; 2,72] | 104,59 % | 144 [141; 144] (23) | 14 [10; 28] |
| 0,46 | 0,488 [0,292; 0,65] | 1,12 [1,03; 1,19] | 12,11 % | 114 [79; 142] (23) | 125 [38; 504] |
| 0,046 | 0,659 [0,514; 0,783] | 1 [1; 1,01] | 0,26 % | 40 [27; 68] (23) | 1 284 [316; 12 274] |

**n = 40** (30 Instanzen):

| ε/OPT | Abstand zum Produktmaß (TV) | Kosten des Plans / OPT | Kostenlücke | Einträge über 1e-6 (Ecke) | Produkte |
|---|---|---|---|---|---|
| 46 | 0,0346 [0,0201; 0,0706] | 5,76 [3,77; 10,3] | 475,68 % | 1 594 [1 534; 1 600] (79) | 8 [8; 10] |
| 4,6 | 0,291 [0,187; 0,469] | 3,34 [2,8; 3,7] | 233,94 % | 1 568 [1 423; 1 598] (79) | 28 [20; 52] |
| 0,46 | 0,726 [0,631; 0,813] | 1,19 [1,15; 1,24] | 18,75 % | 659 [396; 924] (79) | 346 [182; 672] |
| 0,046 | 0,879 [0,858; 0,901] | 1 [1; 1,01] | 0,41 % | 161 [100; 242] (79) | 4 232 [1 966; 12 414] |

Bei ε/OPT = 46 liegt der Plan nahe am Produktmaß: Abstand in Totalvariation 0,0135 (n = 12) und 0,0346 (n = 40) im Median, größter Wert 0,0291 und 0,0706; die Kosten des Plans sind dort das 2,5-Fache (n = 12) und das 5,8-Fache (n = 40) von OPT. **Das Vielfache wächst mit n**: die Kosten des Plans bei großem ε/OPT sind kein festes Vielfaches von OPT. Bei ε/OPT = 0,046 liegt die Kostenlücke bei 0,26 % (n = 12) und 0,41 % (n = 40), der Plan hat aber **noch das 1,7-Fache (n = 12) und das 2,0-Fache (n = 40) der Einträge über 1e-6 der Ecke** (40 gegen 23 und 161 gegen 79): ein kleines ε macht den Plan bei dieser Schwelle und diesem ε/OPT nicht so dünn wie die Ecke (bei kleinerem ε/OPT wird er dünner, wie die Tabellen zeigen; absolut sind es 161 von 1 600 Einträgen bei n = 40). Mit dem Abstand statt des Abstandsquadrats (n = 12) liegt die Lücke bei ε/OPT = 0,46 und 0,046 bei 15,17 % und 0,54 %, die Einträge bei 0,046 bei 59 gegen 23.

### Experiment 2: Die Skala – ε/OPT, nicht ε

Je n 30 Instanzen (quadrierter Abstand); die Lücke bei ε/OPT = 0,07 aus dem Newton-Verfahren. Spalten: LP-Optimum OPT, ε = 0,005 als ε/OPT, Kostenlücke bei ε/OPT = 0,07.

| n | OPT | ε = 0,005 als ε/OPT | Lücke bei ε/OPT = 0,07 |
|---|---|---|---|
| 10 | 0,117 [0,0284; 0,288] | 0,0429 [0,0174; 0,176] | 0,468 [0,065; 1,1] % |
| 20 | 0,0659 [0,034; 0,149] | 0,0759 [0,0335; 0,147] | 0,912 [0,354; 1,7] % |
| 40 | 0,0303 [0,0175; 0,0751] | 0,165 [0,0666; 0,285] | 0,772 [0,345; 1,67] % |
| 80 | 0,0147 [0,0102; 0,0348] | 0,339 [0,144; 0,489] | 0,877 [0,625; 1,95] % |

Ein festes ε = 0,005 ist bei n = 10 gleich 0,043 OPT und bei n = 80 gleich 0,34 OPT. Bei festem ε/OPT = 0,07 liegt die Lücke bei 30 Instanzen je n im Median bei 0,47 % (n = 10), 0,91 % (n = 20), 0,77 % (n = 40) und 0,88 % (n = 80): **bei 30 Instanzen sind die Mediane nicht monoton in n**, und die Streuung zwischen Instanzen gleicher Größe (bei n = 20 von 0,35 % bis 1,7 %) ist größer als der Unterschied zwischen den n. Das ist ein Stichprobeneffekt: in einer Kontrollrechnung mit 100 Instanzen je n (gleiche Erzeugung, gleiche Rechnung) steigt der Median mit n, beim Zufalls-Seed 2026 von 0,49 % (n = 10) über 0,67 % (n = 20) und 0,84 % (n = 40) auf 1,03 % (n = 80), beim Zufalls-Seed 777 von 0,43 % über 0,60 % und 0,82 % auf 1,00 %. Ein Anstieg der Lücke mit n bei festem ε/OPT ist also wahrscheinlich und mit 30 Instanzen nur nicht sicher zu sehen (kein Konfidenzintervall berechnet; `test_kontrolle_experiment_2_mit_100_instanzen_steigt_mit_n`). Dasselbe ε/OPT gibt auch bei Handbeispielen verschiedene Lücken: bei ε/OPT = 0,91 sind es 25,0 % (H1) gegen 0,6 % (H3).

### Experiment 3: Wie viele Produkte, und was die Kosten vor dem Ende zeigen

Rohe Kostenlücke des Plans nach k Iterationen bei ε/OPT = 0,46 (Abbruch ohne Konvergenz, Plan nach dem Spaltenschritt), Median [Minimum; Maximum] über 30 Instanzen, dazu die Zahl der Instanzen mit rohen Kosten unter OPT und der Zeilenfehler (L1, Median).

**n = 12:**

| k | rohe Kostenlücke | Kosten unter OPT | Zeilenfehler |
|---|---|---|---|
| 1 | -29,3 [-55; -4,96] % | 30 von 30 | 0,49 |
| 3 | -4,44 [-34,6; 15,7] % | 17 von 30 | 0,17 |
| 10 | 10,4 [-12,5; 18,8] % | 6 von 30 | 0,022 |
| 30 | 12 [0,436; 18,8] % | 0 von 30 | 8,1e-5 |
| 1000 | 12,1 [2,66; 18,8] % | 0 von 30 | 2,4e-16 |

**n = 40:**

| k | rohe Kostenlücke | Kosten unter OPT | Zeilenfehler |
|---|---|---|---|
| 1 | -41,1 [-57,4; -10,1] % | 30 von 30 | 0,5 |
| 3 | -25,3 [-40,9; 9,97] % | 28 von 30 | 0,28 |
| 10 | 4,24 [-17,8; 17,1] % | 10 von 30 | 0,092 |
| 30 | 17,6 [4,5; 23] % | 0 von 30 | 0,0066 |
| 1000 | 18,7 [14,6; 23,6] % | 0 von 30 | 3,4e-16 |

Bei ε/OPT = 0,46 liegen die Kosten nach **einer Iteration (zwei Produkte)** in 30 von 30 Instanzen (n = 12 und n = 40) unter dem LP-Optimum, im Median 29,3 % (n = 12) und 41,1 % (n = 40); die Zeilensummen sind dann um 0,49 und 0,5 (L1) falsch, die Kosten sind keine Kosten einer Kopplung. Bei anderem ε/OPT hängt es vom ε ab (Preset-Texte der App); im Handbeispiel H3 (ε/OPT = 1,8) sind es genau −20 %. Die Produkte bis Zeilenfehler 1e-8 (zwei je Iteration; Stufen ε/OPT = 1,4 / 0,46 / 0,14 / 0,046 / 0,014, je Stufe ε um den Faktor 3,0 bis 3,3 kleiner):

**n = 12:**

| ε/OPT | Produkte bis Zeilenfehler 1e-8 | nicht konvergiert | Faktor zur Stufe davor (Mediane) |
|---|---|---|---|
| 1,4 | 39 [16; 118] | 0 von 30 | – |
| 0,46 | 125 [38; 504] | 0 von 30 | 3,21 (ε-Faktor 3,04) |
| 0,14 | 502 [102; 1 880] | 0 von 30 | 4,02 (ε-Faktor 3,29) |
| 0,046 | 1 284 [316; 12 274] | 0 von 30 | 2,56 (ε-Faktor 3,04) |
| 0,014 | 3 275 [688; 59 270] | 0 von 30 | 2,55 (ε-Faktor 3,29) |

**n = 40:**

| ε/OPT | Produkte bis Zeilenfehler 1e-8 | nicht konvergiert | Faktor zur Stufe davor (Mediane) |
|---|---|---|---|
| 1,4 | 97 [60; 196] | 0 von 30 | – |
| 0,46 | 346 [182; 672] | 0 von 30 | 3,57 (ε-Faktor 3,04) |
| 0,14 | 1 346 [662; 3 208] | 0 von 30 | 3,89 (ε-Faktor 3,29) |
| 0,046 | 4 232 [1 966; 12 414] | 0 von 30 | 3,14 (ε-Faktor 3,04) |
| 0,014 | 14 918 [3 614; 30 420] | 0 von 30 | 3,53 (ε-Faktor 3,29) |

Die Produkte wachsen bei n = 40 um den Faktor 3,1 bis 3,9 je Stufe von 3,0 bis 3,3 in ε, bei n = 12 um 2,6 bis 4,0 („etwa mit 1/ε“, mit Streuung; bei n = 12 schwächer und streuender). Die Streuung zwischen Instanzen ist groß: bei n = 12 und ε/OPT = 0,014 reicht sie von 688 bis 59 270 Produkten.

### Experiment 4: Glatt statt springend

Pfad $C(t) = C_0 + t\,(C_1 - C_0)$ mit zwei unabhängigen Kostenmatrizen (n = 20, quadrierter Abstand, Massen $0{,}5/n + 0{,}5 \cdot \text{Dirichlet}(1)$, 5 Pfade, Zufalls-Seed 35), ε fest = ε/OPT mal das mittlere LP-Optimum entlang des Pfads. **LP-Plan:** Trägerwechsel (Schritte, in denen sich die Menge der Einträge über 1e-9 ändert; eine Untergrenze der Basiswechsel) bei T = 800 Schritten: 96 bis 110, größter L1-Sprung 0,19 bis 0,35 (Höchstwert 2). **Weicher Plan** (Newton-Verfahren mit Warmstart), größter L1-Sprung je Schritt und ε·L (L = größter L1-Sprung / größte Änderung der Kosten je Schritt), Median [Minimum; Maximum] über die 5 Pfade:

| T | ε/OPT | größter Sprung | ε·L |
|---|---|---|---|
| 50 | 1 | 0,0366 [0,0298; 0,0387] | 0,133 [0,118; 0,162] |
| 50 | 0,1 | 0,174 [0,154; 0,2] | 0,0648 [0,0557; 0,0815] |
| 50 | 0,01 | 0,351 [0,312; 0,394] | 0,0147 [0,00995; 0,0163] |
| 200 | 1 | 0,00914 [0,00746; 0,00969] | 0,133 [0,118; 0,162] |
| 200 | 0,1 | 0,0461 [0,0388; 0,0522] | 0,0667 [0,0563; 0,0846] |
| 200 | 0,01 | 0,135 [0,115; 0,143] | 0,0224 [0,0147; 0,0233] |
| 800 | 1 | 0,00229 [0,00187; 0,00242] | 0,133 [0,118; 0,162] |
| 800 | 0,1 | 0,0117 [0,00976; 0,0131] | 0,067 [0,0566; 0,0848] |
| 800 | 0,01 | 0,0363 [0,0345; 0,0391] | 0,024 [0,0176; 0,0257] |

Der größte Sprung des weichen Plans **sinkt im gemessenen Bereich mit 1/T**: bei vierfacher Schrittzahl (T von 200 auf 800) sinkt er (Verhältnis der Mediane) um den Faktor 4,0 bei ε/OPT = 1, 4,0 bei 0,1 und 3,7 bei 0,01; der Plan ist in dieser Messung also stetig mit endlicher Steigung (bei ε/OPT = 0,01 ist das Raster noch nicht fein gegen ε, daher 3,7 statt 4). In diesen 5 Pfaden wird die Skala 1/ε nicht erreicht: ε·L liegt in allen neun Zeilen der Tabelle unter 0,17 (höchstens 0,162), die Skala 1/ε wird also mindestens um den Faktor 6 unterschritten. **Am Gleichstand wird sie erreicht:** im Handbeispiel 2 × 2 ($C(t) = \begin{pmatrix} t & 1-t \\ 1-t & t \end{pmatrix}$, ε/OPT = 0,1, also ε = 0,05) ist ε·L = 0,95 / 1,00 / 1,00 bei T = 50 / 200 / 800 (genau 1 für T gegen unendlich); der LP-Plan springt dort einmal um 2 (den Höchstwert), der weiche Plan hat bei t = ½ das Produktmaß. Glatt gilt nur, solange die Schrittweite klein gegen ε ist: bei T = 50 und ε/OPT = 0,01 ist der größte Sprung des weichen Plans 0,351 (Median), gegen 0,45 (Median) beim LP-Plan.

### Experiment 5: Wie schnell verschwindet die Verzerrung?

n = 20, 20 Instanzen, ε/OPT von 1 bis 0,0032 in 11 Stufen (Faktor $10^{1/4}$), der Plan aus dem Newton-Verfahren (Randfehler unter 1e-12); Lücken unter 1e-11 sind vom Rechenrauschen nicht zu trennen und stehen nicht in den Tabellen. Steigung = $d \ln \text{Lücke} / d \ln \varepsilon$ der Median-Kurve bis zur nächsten Stufe.

**Stetige Kosten** (Punkte im Quadrat, Massen $0{,}5/n + 0{,}5 \cdot \text{Dirichlet}(1)$; 20 von 20 Instanzen mit eindeutigem Optimum):

| ε/OPT | Median der Kostenlücke | Spanne | Steigung |
|---|---|---|---|
| 1 | 0,46 | 0,37 bis 0,56 | 1,36 |
| 0,56 | 0,21 | 0,16 bis 0,32 | 1,47 |
| 0,32 | 0,0901 | 0,063 bis 0,17 | 1,59 |
| 0,18 | 0,036 | 0,02 bis 0,075 | 1,64 |
| 0,1 | 0,014 | 0,0058 bis 0,032 | 1,78 |
| 0,056 | 0,00504 | 0,0019 bis 0,013 | 1,69 |
| 0,032 | 0,00191 | 0,00053 bis 0,0048 | 1,86 |
| 0,018 | 0,000654 | 0,00017 bis 0,0014 | 2,09 |
| 0,01 | 0,000196 | 2,7e-5 bis 0,00042 | 2,13 |
| 0,0056 | 5,76e-5 | 1,1e-6 bis 0,00021 | 1,69 |
| 0,0032 | 2,17e-5 | 2,6e-9 bis 0,00012 | – |

**Gitterkosten** (6 × 6-Gitter, L1, Massen 1/n; 0 von 20 Instanzen mit eindeutigem Optimum; 61 von 220 (Instanz, ε)-Paaren sind nicht konvergiert und fehlen in den Medianen):

| ε/OPT | Median der Kostenlücke | Spanne | Steigung |
|---|---|---|---|
| 1 | 0,673 | 0,53 bis 0,76 | 1,87 |
| 0,56 | 0,23 | 0,13 bis 0,33 | 2,42 |
| 0,32 | 0,0569 | 0,02 bis 0,11 | 3,12 |
| 0,18 | 0,00946 | 0,00052 bis 0,03 | 4,58 |
| 0,1 | 0,000679 | 4,8e-7 bis 0,0063 | 7,66 |
| 0,056 | 8,25e-6 | 1,9e-12 bis 0,00043 | 14,69 |
| 0,032 | 1,75e-9 | 2,7e-15 bis 4e-6 | – |

Bei eindeutigem Optimum steigt die Steigung der Lückenkurve von 1,4 (ε/OPT von 1 bis 0,56) auf 1,7 bis 2,1 unter ε/OPT = 0,1: **im gemessenen Bereich (ε/OPT 0,0032 bis 1) ist der Verlauf der Median-Kurve eine Potenz mit lokaler Steigung 1,4 bis 2,1, nicht ∝ ε; ein exponentielles Verhalten (eine mit 1/ε wachsende Steigung) ist in diesem Bereich nicht zu erkennen**, in 6 von 20 Instanzen passt die Halblog-Anpassung aber besser als die Log-Log-Anpassung. Wo ein exponentieller Bereich beginnt, ist hier nicht bestimmt. Bei Gitterkosten dagegen wächst die Steigung von 1,9 auf 14,7 und die Median-Lücke (Anteil von OPT) fällt von 0,673 auf 1,75e-9 bei ε/OPT = 0,032 (die Medianen bei kleinem ε/OPT stützen sich wegen der nicht konvergierten Paare auf weniger als 20 Instanzen), und ist im gemessenen Bereich keine Potenz (die Steigung wächst mit jeder Stufe); $\ln$ Lücke gegen OPT/ε ist dort fast linear (Steigung −0,73 bis −0,57): der Verlauf ähnelt einer Exponentialfunktion von 1/ε, **bei Gitterkosten** (ganzzahlige Kosten, 0 von 20 Instanzen mit eindeutigem Optimum, also mehrere optimale Pläne). Ob die Mehrdeutigkeit die Ursache ist, ist nicht getestet (Gitterkosten unterscheiden sich auch durch die ganzzahligen Kosten). Die Wahl des Plans bei Gleichstand und die Konvergenz dort untersucht diese Demo nicht.

### Handbeispiele von Hand gerechnet

**H1** (zwei Lager, zwei Kunden, $a = b = (\tfrac12, \tfrac12)$, Kosten $\begin{pmatrix} 1/2 & 1 \\ 1 & 1/2 \end{pmatrix}$, OPT = ½, Produktmaß kostet ¾): mit ε = 1/(2 ln 3) ist $K = \begin{pmatrix} 1/3 & 1/9 \\ 1/9 & 1/3 \end{pmatrix}$ und nach **einem** Durchlauf $P = \begin{pmatrix} 3/8 & 1/8 \\ 1/8 & 3/8 \end{pmatrix}$ (Kosten ⅝, Kostenlücke 25 %, θ = ½, Abstand zum Produktmaß ¼, ε/OPT = 1/ln 3 ≈ 0,91, vier Einträge gegen zwei der Ecke). Allgemein ist die Lücke $1/(1 + e^{1/e})$ bei ε/OPT = e. **H2:** Kosten und ε mit 10 multipliziert geben denselben Plan, Kosten mal 10 bei gleichem ε die Lücke 1/59 050. **H3** ($a = (\tfrac34, \tfrac14)$, $b = (\tfrac14, \tfrac34)$, Kosten $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, ε/OPT = 2/ln 3 ≈ 1,8): nach einem Durchlauf $P = \begin{pmatrix} 9/40 & 3/8 \\ 1/40 & 3/8 \end{pmatrix}$, Zeilenfehler 3/10, Kosten ⅖ unter OPT = ½ (rohe Lücke −20 %); der Grenzplan hat die Lücke $(3\sqrt 3 - 5)/4$ = +4,9 %. Dasselbe ε wie bei H1 gibt bei H3 nur $(\sqrt{1701} - 41)/40$ = 0,6 % statt 25 %. **H4:** $P_{11}(t) = \tfrac12 \sigma((1 - 2t)/\varepsilon)$, am Gleichstand ε·L = 1.

### Zahl → Test

| Zahl der README | Test |
|---|---|
| Tabellen und Sätze zu Experiment 1 bis 5 (aus `tools/messreihen.json`) | `tests/test_claims.py`: `test_tabellen_der_readme`, `test_saetze_experiment_1` bis `test_saetze_experiment_5` |
| Neu gerechnete Reihen (Band) | `test_vorgerechnete_reihen_stimmen_mit_der_neuen_rechnung_ueberein`, `test_messreihe_n40_bei_eps_46_neu_gerechnet_als_band`, `test_verzerrung_und_pfad_2x2_neu_gerechnet`, `test_kontrolle_experiment_2_neu_gerechnet_als_band` |
| Handbeispiele H1 bis H4 | `tests/test_sinkhorn.py` (Brüche), `tests/test_evaluation.py` |
| Zählung des Abbruchtests (2k + 1 Auswertungen) | `test_abbruchtest_kostet_ein_produkt_mehr_als_gemeldet` (`tests/test_sinkhorn.py`) |
| Orakel-Zahlen, Testzahl | `test_orakel_zahlen`, `test_testzahl_der_readme` |

Die Reihen mit $n = 40$ und $T = 800$ stehen in der Datei und werden nicht jedes Mal neu gerechnet; die kleinen Reihen rechnet der Test neu und vergleicht als Band, weil Gleitkomma-Summen und die Wahl der Ecke bei Gleichstand auf anderen Plattformen abweichen können.

## Ehrliche Grenzen

- **Gültigkeit:** Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Instanz; Matrix-Vektor-Produkte). Ein Median über 30 Instanzen (bei Experiment 4 über 5 Pfade) ist kein Gesetz; andere Zufalls-Seeds geben andere Zahlen im Band; eine Folgerung aus 30 Instanzen kann kippen (Experiment 2: der Anstieg mit n ist erst mit 100 Instanzen je n zu sehen).
- **Annahmen, die hier stehen bleiben:** (1) *Der Kern K = exp(−C/ε) ist überall positiv, beide Randverteilungen haben dieselbe Gesamtmasse, ε ist ein frei gewählter Regler, und „gelaufen“ heißt „gelöst“.* Nach einer Iteration sind die Zeilensummen falsch (Experiment 3); Nullen im Kern, verschiedene Gesamtmassen und ein aus Beobachtungen zurückgerechnetes ε sind nicht untersucht. (2) *Der gezeigte Plan ist exakt gerechnet: kein Unterlauf, kein Gleichstand, und seine Ableitung ist die des Grenzplans.* Die Demo rechnet in float64 in der Log-Domäne; endliche Genauigkeit, Unterlauf, die Wahl bei Gleichstand und die Ableitung am unfertigen Plan sind nicht untersucht. (3) *⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.* Sie wird besser (Experiment 5), ist aber auch dann kein Abstand (für gleiche Verteilungen nicht 0); die Demo benutzt das Wort „Sinkhorn-Distanz“ nicht, weil es mehrere Definitionen gibt.
- **Nicht gemessen:** Verteilungen mit Atomen (Zonenmatrizen), nicht quadrierte Abstände bei Glätte und Steigung (Experiment 4 und 5 nehmen den quadrierten Abstand beziehungsweise Gitterkosten; die Kostenart der Seitenleiste wirkt auf Experiment 4), n ≠ m, Gesamtmassen ungleich 1; der Anstieg der Lücke mit n bei festem ε/OPT: bei 30 Instanzen nicht monoton, mit 100 Instanzen je n steigend (Experiment 2), über n = 80 hinaus nicht untersucht; die Schwelle, ab der der Verlauf bei eindeutigem Optimum exponentiell wird.
- **Newton statt Iteration bei sehr kleinem ε:** Experimente 2 (ε/OPT = 0,07), 4 und 5 benutzen das Newton-Verfahren; ob die reine Iteration dort in 30 000 Iterationen ausreicht, ist nicht gemessen; die App meldet „nicht konvergiert“ beim Limit.
- **Limit der Iteration:** In den Produktreihen von n = 12 und n = 40 (300 Läufe) erreichte kein Lauf das Limit von 30 000 Iterationen; bei noch kleinerem ε/OPT ist das nicht ausgeschlossen, die App meldet es als „nicht konvergiert“.
- **Nicht mit echten Mausklicks im Browser durchgeklickt:** AppTest und Tests sehen die Plotly-Abbildungen nicht.
- **Literatur:** Cuturi (2013), arXiv 1306.0895 (Gl. (2), (3); arXiv-Fassung gelesen); Altschuler, Weed, Rigollet (2017), arXiv 1705.09634 (Skalierungsform, Lemma 1); Benamou, Carlier, Cuturi, Nenna, Peyré (2015), arXiv 1412.5154 (Gl. (6), KL-Projektion); Nutz, Introduction to Entropic Optimal Transport (Vorlesungsskript 2022, Abschn. 4; Schrödinger-Brücke; Gleichungsnummern hier nicht nachgeprüft); Léonard (2014), arXiv 1308.0215 (Γ-Limes zum quadratischen Monge-Kantorovich-Problem, Abschn. 5 laut Gliederung; Wortlaut nicht abschließend geprüft); Peyré, Cuturi (2019), Computational Optimal Transport (Hintergrund, Kapitel 1 bis 4; keine Zahl der Demo hängt an ihm). Alles ist Lehrbuch oder Forschung; die Demo zählt und zeigt es an Instanzen. Dieselbe Rechnung heißt in der Verkehrsplanung Furness-Verfahren.

## Verifikation

- **Kern:** Handbeispiele H1 bis H6 in Brüchen und in Zahlen (`tests/test_sinkhorn.py`): Plan nach einem Durchlauf, rohe Lücke −20 %, geschlossene Form der Lücke, Skalierung der Einheit, Zählung (7 Iterationen = 14 Produkte), Grenzplan; Einheiten (Log-Sum-Exp, f- und g-Schritt, Abbruch, Warmstart, Randfälle n = 1, gleiche Kosten, sehr kleines ε).
- **Orakel** (`tests/test_oracle_sinkhorn.py`, ein anderer Rechenweg): der weiche Plan gegen das **Dual mit L-BFGS** (20 Zufallsinstanzen bei zwei Werten von ε/OPT, Toleranz 2e-7), gegen die **naive Iteration im K-Bereich** nach jedem k (12 Instanzen, 7 Werte von k, relative Toleranz 1e-9), gegen die **KL-Projektion** mit Nelder-Mead (8 Instanzen, n = m = 3) und gegen das **Newton-Verfahren**; die 2 × 2-Formel $x(t)$; das LP-Optimum gegen `linprog` (HiGHS, 40 Instanzen mit n ≠ m), die inneren Punkte (8 Instanzen), `networkx.network_simplex` und den Mindestkostenfluss von OR-Tools (15 ganzzahlige Instanzen, Abweichung 0); der Eindeutigkeitstest gegen ein zweites LP auf der optimalen Fläche (25 Instanzen). Dazu ein großer Lauf außerhalb des Repos mit 200 Zufallsinstanzen (Sinkhorn gegen L-BFGS-Dual, Newton und die naive Iteration, LP gegen `linprog`).
- **Eigenschaften** (`tests/test_eigenschaften.py`, `tests/test_newton.py`): Kosten und ε mit demselben Faktor, Umnummerieren, Vertauschen von Angebot und Nachfrage, ε gegen unendlich gibt das Produktmaß, $0 \le \langle C, P_\varepsilon\rangle - \mathrm{OPT} \le \varepsilon (h(a) + h(b))$, die Kosten wachsen mit ε, der Plan ist dicht, θ fällt mit kleinerem ε und der Abstand zum Produktmaß (TV) wächst mit kleinerem ε, Konvergenz des Newton-Verfahrens bis ε/OPT = 0,003.
- **Verdikte und Zahlen:** Verdikte sind reine Funktionen (`snk_evaluation.py`) und an handgesetzten Zeilen geprüft (`tests/test_evaluation.py`); jede Zahl dieser README steht in `tests/test_claims.py`.
- **Oberfläche:** AppTest (jeder Button, jeder Regler an Minimum und Maximum, die Optionen der Auswahlregler, jedes Preset mit seiner erwarteten Aussage, Permalink, Worst-Case am Regler-Maximum), Layout- und Footer-Test.
- **Kopien:** `snk_simplex.py` und `snk_eindeutigkeit.py` sind Kopien der Wurzel-Demo, `snk_rng.py` der Vorlage des Portfolios (Kopier-Tests).
- **Testzahl:** 187 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und die Bibliotheken der Gegenproben (scipy, networkx, OR-Tools) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly |
| `tools/messreihen.py` | rechnet die Messreihen der README neu und schreibt `tools/messreihen.json` |
| `tools/vorrechnen.py` | rechnet die Reihen für die Standardeinstellungen vor und schreibt `tools/vorgerechnet.json` |
| `snk_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte |
| `snk_eindeutigkeit.py` | exakter Eindeutigkeitstest für den LP-Plan (Kopie der Wurzel-Demo) |
| `snk_evaluation.py` | Kennzahlen, Reihen der fünf Experimente, Verdikte |
| `snk_layout.py` | Fester Kopf (Stylesheet) |
| `snk_newton.py` | Newton-Verfahren auf dem Dual mit Fortsetzung in ε |
| `snk_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `snk_rng.py` | Hausgenerator SplitMix64 (plattformgleich) |
| `snk_scenario.py` | Instanzen: Massen, Kosten, Gitter, Pfade, Handbeispiele |
| `snk_simplex.py` | Transportsimplex mit Basisbaum (Kopie der Wurzel-Demo) |
| `snk_sinkhorn.py` | Sinkhorn-Iteration in der Log-Domäne (wiederverwendbar, ohne Streamlit) |
| `snk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `snk_vorgerechnet.py` | liest die vorgerechneten Reihen |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Newton, Szenario und Kopien, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Kein Vergleich mit dem Simplex nach Rechenschritten oder Zeit:** die Demo zählt Matrix-Vektor-Produkte der Iteration und sagt nichts darüber, ob sie schneller als ein anderes Verfahren ist.
- **Keine Bibliothek zur Laufzeit außer numpy und plotly** (`scipy`, `networkx`, OR-Tools nur in den Tests).
- **Keine ungleichen Gesamtmassen, keine Nullen im Kern, kein n ≠ m in der App** (der Löser kann n ≠ m, die Regler bieten es nicht an).
- **Kein PDF-Export.**
- **Keine Wiederholung der Mitte-der-Fläche-Rechnung bei Gleichstand:** hier nur die Lückenkurve (Experiment 5) und das Handbeispiel.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
