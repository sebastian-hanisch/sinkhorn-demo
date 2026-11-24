"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen von Zellen und Pfeilen sind Annotationen (kein Text-Trace, keine texttemplate)."""

import math

import numpy as np
import plotly.graph_objects as go

import snk_constants as C

MAX_ZELLEN_MIT_ZAHL = 144      # bis 12 x 12 steht die Zahl in der Zelle
SCHWELLE_PFEIL = 1e-3          # Pfeile der Karte: Einträge über diesem Anteil der Gesamtmasse
LOG_BODEN = 1e-12


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def zahl_text(x: float, gesamt: float) -> str:
    """Zellzahl: Anteil an der Gesamtmasse in Prozent mit Komma (unter 0,05 % als „< 0,1“)."""
    p = 100.0 * x / gesamt
    if p < 0.05:
        return "&lt;0,1"
    return f"{p:.1f}".replace(".", ",")


def build_plan_matrix(plan, a, b, titel="", hoehe=380):
    """Der Plan als Matrix (Zeilen = Lager, Spalten = Kunden, Farbe und Zahl = Anteil der Gesamtmasse in Prozent); Randverteilungen in den Achsenbeschriftungen.
    Einträge unter 1e-6 bleiben leer (das ist die Zählgrenze der Kennzahl „Einträge über 1e-6“)."""
    plan = np.asarray(plan, float)
    n, m = plan.shape
    gesamt = float(plan.sum())
    z = np.where(plan > C.SCHWELLE_EINTRAG, 100.0 * plan / gesamt, np.nan)
    xs = [f"K{j + 1}" for j in range(m)]
    ys = [f"L{i + 1}" for i in range(n)]
    zmax = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale=[[0, "#e3efed"], [1, C.COLOR_LAGER]], showscale=False, xgap=1, ygap=1, zmin=0, zmax=zmax,
                               hovertemplate="%{y} → %{x}: %{z:.3f} %<extra></extra>"))
    if n * m <= MAX_ZELLEN_MIT_ZAHL:
        beschriftungen = []        # alle Annotationen auf einmal setzen (add_annotation prüft jedes Mal die ganze Liste: quadratisch langsam)
        for i in range(n):
            for j in range(m):
                if plan[i, j] > C.SCHWELLE_EINTRAG:
                    hell = 100.0 * plan[i, j] / gesamt > 0.55 * zmax
                    beschriftungen.append(dict(x=xs[j], y=ys[i], text=zahl_text(plan[i, j], gesamt), showarrow=False,
                                               font=dict(size=11 if n <= 4 else 8, color="white" if hell else "#1b3b37")))
        fig.update_layout(annotations=beschriftungen)
    fig.update_xaxes(side="top", tickmode="array", tickvals=xs, ticktext=[f"K{j + 1}<br>{100.0 * b[j]:.1f}".replace(".", ",") if m <= 12 else f"K{j + 1}" for j in range(m)],
                     tickfont=dict(size=9))
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=ys, ticktext=[f"L{i + 1} · {100.0 * a[i]:.1f}".replace(".", ",") if n <= 12 else f"L{i + 1}" for i in range(n)],
                     tickfont=dict(size=9))
    fig = _base(fig, height=hoehe, legend=False)
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)), margin=dict(l=10, r=10, t=95, b=10))
    return fig


def build_plan_karte(lager, kunden, plan, titel="", hoehe=380, farbe=None):
    """Die Orte (Lager = Quadrate, Kunden = Kreise) und der Plan als Pfeile; die Breite wächst mit dem Eintrag. Gezeichnet werden Einträge über 0,1 % der Gesamtmasse."""
    plan = np.asarray(plan, float)
    n, m = plan.shape
    gesamt = float(plan.sum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lager[:, 0], y=lager[:, 1], mode="markers+text" if n <= 12 else "markers", name="Lager", text=[f"L{i + 1}" for i in range(n)],
                             textposition="top center", marker=dict(symbol="square", size=11, color=C.COLOR_LAGER), hovertemplate="Lager %{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=kunden[:, 0], y=kunden[:, 1], mode="markers+text" if m <= 12 else "markers", name="Kunden", text=[f"K{j + 1}" for j in range(m)],
                             textposition="bottom center", marker=dict(symbol="circle", size=11, color=C.COLOR_KUNDE), hovertemplate="Kunde %{text}<extra></extra>"))
    groesste = float(plan.max()) if plan.size else 1.0
    pfeile = []
    for i in range(n):
        for j in range(m):
            if plan[i, j] > SCHWELLE_PFEIL * gesamt:
                pfeile.append(dict(x=kunden[j, 0], y=kunden[j, 1], ax=lager[i, 0], ay=lager[i, 1], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=2, arrowsize=1.0, arrowwidth=0.5 + 4.5 * plan[i, j] / groesste, arrowcolor=farbe or C.COLOR_PFEIL, standoff=5))
    fig.update_layout(annotations=pfeile)
    sx = [min(lager[:, 0].min(), kunden[:, 0].min()), max(lager[:, 0].max(), kunden[:, 0].max())]
    sy = [min(lager[:, 1].min(), kunden[:, 1].min()), max(lager[:, 1].max(), kunden[:, 1].max())]
    rand = 0.08 * max(sx[1] - sx[0], sy[1] - sy[0], 1e-9)
    fig.add_trace(go.Scatter(x=[sx[0] - rand, sx[1] + rand], y=[sy[0] - rand, sy[1] + rand], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x")
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig, height=hoehe)


def _log_achse(fig, which, lo, hi, titel):
    ax = dict(type="log", range=[math.log10(lo), math.log10(hi)], title_text=titel)
    if which == "x":
        fig.update_xaxes(**ax)
    else:
        fig.update_yaxes(**ax)


def build_eps_reihe(zeilen):
    """Experiment 1: TV zum Produktmaß und Kosten/OPT (Median mit Spanne) gegen ε/OPT. Zwei Diagramme nebeneinander (Gesamtbild in der App)."""
    xs = [z["eps_rel"] for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[z["fach"][0] for z in zeilen], mode="lines+markers", name="Kosten des Plans / OPT (Median)", line=dict(color=C.COLOR_LINE),
                             error_y=dict(type="data", symmetric=False, array=[z["fach"][2] - z["fach"][0] for z in zeilen], arrayminus=[z["fach"][0] - z["fach"][1] for z in zeilen],
                                          color="rgba(76,120,168,0.5)"), hovertemplate="ε/OPT = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(128,128,128,0.8)", annotation_text="LP-Optimum", annotation_position="bottom right")
    fig.update_xaxes(type="log", autorange="reversed", title_text="ε/OPT (links weich, rechts scharf)", tickvals=xs, ticktext=[f"{x:g}".replace(".", ",") for x in xs])
    fig.update_yaxes(type="log", title_text="Kosten des Plans in Vielfachen von OPT")
    return _base(fig, height=320, legend=False)


def build_roh_verlauf(roh, einzel=None, k_marke=None):
    """Experiment 3: rohe Kostenlücke (Median, Spanne über die Instanzen) gegen die Zahl der Iterationen k, mit Null-Linie; Zeilenfehler als zweites Diagramm."""
    k = roh["k"]
    med = [100 * x[0] for x in roh["luecke"]]
    lo = [100 * x[1] for x in roh["luecke"]]
    hi = [100 * x[2] for x in roh["luecke"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="Spanne über die Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=med, mode="lines+markers", name="Median", line=dict(color=C.COLOR_LINE), hovertemplate="k = %{x}: %{y:.2f} %<extra></extra>"))
    if einzel is not None:
        fig.add_trace(go.Scatter(x=einzel[0], y=[100 * x for x in einzel[1]], mode="lines+markers", name="Instanz der Seitenleiste", line=dict(color=C.COLOR_LP, dash="dot"),
                                 hovertemplate="k = %{x}: %{y:.2f} %<extra></extra>"))
    fig.add_hline(y=0.0, line_color="rgba(128,128,128,0.9)", annotation_text="LP-Optimum (Lücke 0)", annotation_position="top right")
    if k_marke is not None:
        fig.add_vline(x=k_marke, line_dash="dash", line_color="rgba(192,80,77,0.8)")
    fig.update_xaxes(type="log", title_text="Iterationen k", tickvals=k, ticktext=[str(x) for x in k])
    fig.update_yaxes(title_text="rohe Kostenlücke (%)")
    return _base(fig, height=330)


def build_zeilenfehler(roh, k_marke=None):
    k = roh["k"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=[max(x[2], LOG_BODEN) for x in roh["zeilenfehler"]], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=[max(x[1], LOG_BODEN) for x in roh["zeilenfehler"]], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND,
                             name="Spanne über die Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=[max(x[0], LOG_BODEN) for x in roh["zeilenfehler"]], mode="lines+markers", name="Median", line=dict(color=C.COLOR_LINE),
                             hovertemplate="k = %{x}: %{y:.2e}<extra></extra>"))
    if k_marke is not None:
        fig.add_vline(x=k_marke, line_dash="dash", line_color="rgba(192,80,77,0.8)")
    fig.update_xaxes(type="log", title_text="Iterationen k", tickvals=k, ticktext=[str(x) for x in k])
    fig.update_yaxes(type="log", title_text="Zeilenfehler (L1)", range=[-8, 0])
    return _base(fig, height=330)


def build_produkte(prod):
    """Experiment 3: Matrix-Vektor-Produkte bis Zeilenfehler 1e-8 gegen ε/OPT (Median, Spanne) mit der Hilfsgeraden ∝ 1/ε."""
    zeilen = prod["zeilen"]
    xs = [z["eps_rel"] for z in zeilen]
    med = [z["produkte"][0] for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=med, mode="lines+markers", name="Median", line=dict(color=C.COLOR_LINE),
                             error_y=dict(type="data", symmetric=False, array=[z["produkte"][2] - z["produkte"][0] for z in zeilen], arrayminus=[z["produkte"][0] - z["produkte"][1] for z in zeilen],
                                          color="rgba(76,120,168,0.5)"), hovertemplate="ε/OPT = %{x}: %{y:.0f} Produkte<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[med[0] * xs[0] / x for x in xs], mode="lines", name="∝ 1/ε (Hilfsgerade durch die erste Stufe)", line=dict(color="rgba(128,128,128,0.9)", dash="dot"),
                             hoverinfo="skip"))
    fig.update_xaxes(type="log", autorange="reversed", title_text="ε/OPT", tickvals=xs, ticktext=[f"{x:g}".replace(".", ",") for x in xs])
    fig.update_yaxes(type="log", title_text="Matrix-Vektor-Produkte bis Zeilenfehler 1e-8")
    return _base(fig, height=330)


def build_skala(zeilen, eps_roh, eps_rel_fest):
    """Experiment 2: links ε/OPT eines festen ε je Instanz (Punkte) und Median je n; rechts die Lücke bei festem ε/OPT je Instanz."""
    ns = [z["n"] for z in zeilen]
    pos = {n: i for i, n in enumerate(ns)}
    fig1, fig2 = go.Figure(), go.Figure()
    for fig, schluessel, faktor, name in ((fig1, "eps_rel_liste", 1.0, "Instanz"), (fig2, "luecke_liste", 100.0, "Instanz")):
        xs, ys = [], []
        for z in zeilen:
            werte = [w for w in z[schluessel] if w == w]
            ver = np.linspace(-0.18, 0.18, len(werte)) if len(werte) > 1 else [0.0]
            xs += [pos[z["n"]] + d for d in ver]
            ys += [faktor * w for w in werte]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=name, marker=dict(size=6, color="rgba(76,120,168,0.55)"), hovertemplate="%{y:.3g}<extra></extra>"))
    fig1.add_trace(go.Scatter(x=list(range(len(ns))), y=[z["eps_rel_fest"][0] for z in zeilen], mode="lines+markers", name="Median", line=dict(color=C.COLOR_LP)))
    fig1.update_yaxes(type="log", title_text=f"ε/OPT bei ε = {eps_roh:g}".replace(".", ","))
    mit = [z for z in zeilen if z["luecke"]]
    fig2.add_trace(go.Scatter(x=[pos[z["n"]] for z in mit], y=[100 * z["luecke"][0] for z in mit], mode="lines+markers", name="Median", line=dict(color=C.COLOR_LP)))
    fig2.update_yaxes(title_text=f"Kostenlücke bei ε/OPT = {eps_rel_fest:g} (%)".replace(".", ","), rangemode="tozero")
    for fig in (fig1, fig2):
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(ns))), ticktext=[f"n = {n}" for n in ns], range=[-0.5, len(ns) - 0.5])
    return _base(fig1, height=320), _base(fig2, height=320)


def build_pfad(ergebnis, zwei_mal_zwei=False, t_marke=None):
    """Experiment 4: L1-Sprung je Schritt entlang des Pfads: LP-Plan (Punkte, nur Schritte mit Sprung) und weiche Pläne (Linien) auf logarithmischer Achse."""
    T = ergebnis["T"]
    ts = [(k + 0.5) / T for k in range(T)]
    fig = go.Figure()
    sp = ergebnis["lp_spruenge"]
    fig.add_trace(go.Scatter(x=[t for t, s in zip(ts, sp) if s > 1e-9], y=[s for s in sp if s > 1e-9], mode="markers", name="LP-Plan", marker=dict(color=C.COLOR_LP, size=7),
                             hovertemplate="t = %{x:.3f}: Sprung %{y:.3f}<extra></extra>"))
    farben = ["#4c78a8", "#2F6B65", "#8e6bbf", "#c0504d"]
    for w, farbe in zip([w for w in ergebnis["weich"] if w.get("ok")], farben):
        fig.add_trace(go.Scatter(x=ts, y=[max(s, LOG_BODEN) for s in w["spruenge"]], mode="lines", name=f"weich, ε/OPT = {w['eps_rel']:g}".replace(".", ","), line=dict(color=farbe),
                                 hovertemplate="t = %{x:.3f}: Sprung %{y:.4f}<extra></extra>"))
    if t_marke is not None:
        fig.add_vline(x=t_marke, line_dash="dash", line_color="rgba(128,128,128,0.8)")
    fig.update_xaxes(range=[0, 1], title_text="Pfadwert t")
    fig.update_yaxes(type="log", range=[-4.5, math.log10(2.2)], title_text="L1-Sprung des Plans je Schritt")
    return _base(fig, height=340)


def build_verzerrung(r):
    """Experiment 5: Kostenlücke gegen ε/OPT auf logarithmischen Achsen (Median, Spanne), Hilfsgeraden der Steigungen 1 und 2 durch den Median bei ε/OPT = 0,56."""
    st = r["stufen"]
    med = [m if m == m and m > 1e-11 else None for m in r["median"]]
    lo = [m if m == m and m > 1e-11 else None for m in r["minimum"]]
    hi = [m if m == m and m > 1e-11 else None for m in r["maximum"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=st, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="Spanne über die Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=st, y=med, mode="lines+markers", name="Median", line=dict(color=C.COLOR_LINE), hovertemplate="ε/OPT = %{x:.3g}: Lücke %{y:.3e}<extra></extra>"))
    j0 = min(range(len(st)), key=lambda j: abs(st[j] - 0.56))
    if med[j0]:
        for steigung, name in ((1, "Steigung 1"), (2, "Steigung 2")):
            fig.add_trace(go.Scatter(x=st, y=[med[j0] * (e / st[j0]) ** steigung for e in st], mode="lines", name=name, line=dict(color="rgba(128,128,128,0.9)", dash="dot" if steigung == 1 else "dash"),
                                     hoverinfo="skip"))
    fig.update_xaxes(type="log", title_text="ε/OPT", range=[math.log10(min(st)) - 0.1, 0.1])
    fig.update_yaxes(type="log", title_text="Kostenlücke (⟨C, P⟩ − OPT) / OPT", range=[-11, 0.3])
    return _base(fig, height=360)
