"""charts.py — All matplotlib chart generators for VoiceIQ Pro"""

import io, base64, json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

PALETTE = {"positive":"#22c55e","neutral":"#f59e0b","negative":"#ef4444",
           "HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}
EMOTION_COLORS = {"angry":"#ef4444","frustrated":"#f97316","anxious":"#f59e0b",
                  "confused":"#a78bfa","neutral":"#64748b","happy":"#22c55e","satisfied":"#10b981"}
BG, BG2, BORDER, TEXT, MUTED = "#0d1117","#161b22","#30363d","#e6edf3","#8b949e"

def _fig(w=6, h=3.5, n=1):
    if n == 1:
        fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
        ax.set_facecolor(BG2)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        return fig, ax
    fig, axes = plt.subplots(1, n, figsize=(w, h), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG2)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.title.set_color(TEXT)
    return fig, axes

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=BG)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def chart_sentiment_donut(df):
    counts = df["sentiment"].value_counts()
    labels, sizes = counts.index.tolist(), counts.values.tolist()
    colors = [PALETTE.get(l,"#64748b") for l in labels]
    fig, ax = plt.subplots(figsize=(4.2, 4), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, _, autotexts = ax.pie(sizes, labels=None, colors=colors, startangle=90,
        autopct="%1.0f%%", pctdistance=0.78,
        wedgeprops={"width":0.52,"edgecolor":BG,"linewidth":2})
    for at in autotexts: at.set_color("white"); at.set_fontsize(11); at.set_fontweight("bold")
    patches = [mpatches.Patch(color=c, label=l.title()) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="center", frameon=False, labelcolor=TEXT, fontsize=10)
    ax.set_title("Sentiment Split", color=TEXT, fontsize=12, pad=6)
    return _b64(fig)

def chart_topic_bar(df):
    counts = df["topic"].value_counts().head(8)
    fig, ax = _fig(7, 4)
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#6366f1", height=0.58)
    for bar in bars:
        ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                str(int(bar.get_width())), va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("Conversations", color=MUTED)
    ax.set_title("Top Customer Issues", color=TEXT, fontsize=12)
    ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    return _b64(fig)

def chart_emotion_bar(df):
    counts = df["emotion"].value_counts()
    colors = [EMOTION_COLORS.get(e,"#64748b") for e in counts.index]
    fig, ax = _fig(7, 3.5)
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.6)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(int(bar.get_height())), ha="center", color=TEXT, fontsize=9)
    ax.set_title("Customer Emotions", color=TEXT, fontsize=12)
    ax.grid(axis="y", color=BORDER, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    return _b64(fig)

def chart_escalation(df):
    counts = df["escalation_level"].value_counts().reindex(["HIGH","MEDIUM","LOW"], fill_value=0)
    colors = [PALETTE[k] for k in counts.index]
    fig, ax = _fig(5, 3.5)
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(int(bar.get_height())), ha="center", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_title("Escalation Risk Levels", color=TEXT, fontsize=12)
    ax.grid(axis="y", color=BORDER, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    return _b64(fig)

def chart_trend(df):
    import pandas as pd
    if "date" not in df.columns: return None
    df2 = df.copy(); df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"])
    if df2.empty: return None
    df2["week"] = df2["date"].dt.to_period("W")
    weekly = df2.groupby(["week","sentiment"]).size().unstack(fill_value=0)
    weekly.index = [str(w) for w in weekly.index]
    fig, ax = _fig(8, 3.5)
    for sent, color in [("negative","#ef4444"),("positive","#22c55e"),("neutral","#f59e0b")]:
        if sent in weekly.columns:
            ax.plot(range(len(weekly)), weekly[sent], color=color, linewidth=2,
                    marker="o", markersize=4, label=sent.title())
    ax.set_xticks(range(len(weekly)))
    ax.set_xticklabels([w.split("/")[0] for w in weekly.index], rotation=35, fontsize=7)
    ax.set_title("Weekly Sentiment Trend", color=TEXT, fontsize=12)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    ax.grid(color=BORDER, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    return _b64(fig)

def chart_agent(df):
    if "agent" not in df.columns or "csat_score" not in df.columns: return None
    scores = df.groupby("agent")["csat_score"].mean().sort_values()
    fig, ax = _fig(6, 3.5)
    colors = ["#22c55e" if s >= 3.5 else "#f59e0b" if s >= 2.5 else "#ef4444" for s in scores]
    bars = ax.barh(scores.index, scores.values, color=colors, height=0.5)
    for bar in bars:
        ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.1f}", va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, 5.5); ax.set_xlabel("Avg CSAT (out of 5)", color=MUTED)
    ax.set_title("Agent CSAT Performance", color=TEXT, fontsize=12)
    ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    return _b64(fig)

def chart_absa(df):
    """Aspect-Based Sentiment Analysis radar/bar chart."""
    import json as _json
    aspect_counts = Counter()
    aspect_neg    = Counter()
    for row in df.get("aspects", []):
        try:
            aspects = _json.loads(row) if isinstance(row, str) else row
            for a in aspects:
                aspect_counts[a["aspect"]] += 1
                if a["sentiment"] == "negative":
                    aspect_neg[a["aspect"]] += 1
        except: pass
    if not aspect_counts: return None
    labels = list(aspect_counts.keys())
    neg_pcts = [aspect_neg[l]/aspect_counts[l]*100 for l in labels]
    fig, ax = _fig(7, 3.8)
    colors = ["#ef4444" if p > 50 else "#f59e0b" if p > 25 else "#22c55e" for p in neg_pcts]
    bars = ax.barh(labels[::-1], neg_pcts[::-1], color=colors[::-1], height=0.55)
    for bar in bars:
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.0f}%", va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, 110); ax.set_xlabel("% Negative Sentiment", color=MUTED)
    ax.set_title("Aspect-Based Sentiment (What's causing dissatisfaction?)", color=TEXT, fontsize=11)
    ax.axvline(50, color=BORDER, linestyle="--", alpha=0.6)
    ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    return _b64(fig)

def build_agent_scorecard(df):
    if "agent" not in df.columns: return []
    rows = []
    for agent in sorted(df["agent"].dropna().unique()):
        sub = df[df["agent"]==agent]; n = len(sub)
        neg_pct  = (sub["sentiment"]=="negative").mean()*100
        high_esc = (sub["escalation_level"]=="HIGH").mean()*100
        avg_csat = sub["csat_score"].mean() if "csat_score" in df.columns else None
        res_rate = sub["resolved"].mean()*100 if "resolved" in df.columns else None
        perf = 0
        if avg_csat: perf += (avg_csat/5)*40
        if res_rate: perf += (res_rate/100)*35
        perf += ((100-high_esc)/100)*25
        rows.append({"agent":agent,"interactions":n,
            "avg_csat":f"{avg_csat:.1f}" if avg_csat else "N/A",
            "resolution_rate":f"{res_rate:.0f}%" if res_rate else "N/A",
            "escalation_rate":f"{high_esc:.0f}%",
            "neg_sentiment":f"{neg_pct:.0f}%",
            "performance_score":int(perf)})
    return sorted(rows, key=lambda r: r["performance_score"], reverse=True)
