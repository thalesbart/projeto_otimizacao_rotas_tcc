"""Figuras 1, 2, 4 e 5 do TCC a partir das saidas da v11 (saidas_v11)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
BASE = sys.argv[1] if len(sys.argv) > 1 else "saidas_v11"
T = f"{BASE}/tabelas/"
CINZA, VERDE, LARANJA, TINTA = "#858c95", "#1f7f7a", "#d9822b", "#222222"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
br = lambda v, d=1: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
fmt = lambda d: FuncFormatter(lambda v, _: br(v, d))

# ---- Figura 1: funcao objetivo media por cenario
r = pd.read_excel(T + "resumo_ablation_ic95.xlsx").sort_values("funcao_objetivo_media")
rot = {"Balanceamento + ACO + 2OPT": "Balanceamento\n+ ACO + 2OPT", "Balanceamento + ACO sem 2OPT": "Balanceamento\n+ ACO sem 2OPT",
       "Balanceamento + Nearest Neighbor": "Balanceamento\n+ Nearest\nNeighbor", "Balanceamento + FIFO": "Balanceamento\n+ FIFO",
       "Sem balanceamento + FIFO": "Sem\nbalanceamento\n+ FIFO\n(referência)"}
fig, ax = plt.subplots(figsize=(9.2, 5.6))
cores = [VERDE if c == "Balanceamento + ACO + 2OPT" else CINZA for c in r.cenario]
ax.bar([rot[c] for c in r.cenario], r.funcao_objetivo_media, yerr=r.funcao_objetivo_ic95, color=cores, capsize=4, width=0.78, error_kw={"ecolor": TINTA, "lw": 1})
for i, v in enumerate(r.funcao_objetivo_media):
    ax.text(i, v + r.funcao_objetivo_media.max() * 0.015, br(v, 0), ha="center", va="bottom", fontweight="bold", color=TINTA, fontsize=13)
ax.set_ylabel("Função objetivo média (un.)"); ax.yaxis.set_major_formatter(fmt(0))
ax.set_ylim(0, r.funcao_objetivo_media.max() * 1.12)
ax.set_title(f"Função objetivo média por cenário (médias de {int(r.dias.iloc[0])} dias, IC95%)", fontweight="bold")
fig.tight_layout(); fig.savefig("figura1_funcao_objetivo.png", dpi=200); plt.close(fig)

# ---- Figura 2: boxplot da carga por equipe
c = pd.read_excel(T + "cargas_equipes_boxplot.xlsx")
ordem = ["Sem balanceamento", "Com balanceamento"]
dados = [c.loc[c.cenario == k, "carga_horas"].values for k in ordem]
medias = [c[c.cenario == k].groupby("dia").carga_horas.mean().mean() for k in ordem]
fig, eixos = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, lim, tit in ((eixos[0], None, "Distribuição completa"), (eixos[1], (7.0, 8.05), "Ampliação: 7 a 8 horas")):
    bx = ax.boxplot(dados, showfliers=True, patch_artist=True, widths=0.5,
                    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35, "markerfacecolor": "#555555", "markeredgecolor": "none"},
                    medianprops={"color": TINTA, "linewidth": 1.8})
    for caixa, k in zip(bx["boxes"], ordem): caixa.set_facecolor(CINZA if k.startswith("Sem") else VERDE)
    ax.scatter([1, 2], medias, marker="D", s=60, color=LARANJA, zorder=4, label="Média (média das médias diárias)")
    for x, m in zip([1, 2], medias):
        ax.text(x + 0.30, m, br(m, 2) + " h", va="center", ha="left", color=LARANJA, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xticks([1, 2]); ax.set_xticklabels(ordem); ax.set_xlim(0.5, 2.8)
    ax.yaxis.set_major_formatter(fmt(1)); ax.set_ylabel("Carga por equipe (horas)"); ax.set_title(tit)
    if lim: ax.set_ylim(*lim)
fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", frameon=False, fontsize=10)
fig.suptitle("Carga por equipe, sem e com balanceamento", fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("figura2_boxplot_carga.png", dpi=200); plt.close(fig)

# ---- Figura 4: 2OPT antes e depois
d = pd.read_excel(T + "comparativo_antes_depois_2opt.xlsx")
a, b_ = d.antes_2opt_km.mean(), d.depois_2opt_km.mean(); q = (1 - b_ / a) * 100
fig, ax = plt.subplots(figsize=(6, 6))
ax.bar(["Antes do 2OPT", "Depois do 2OPT"], [a, b_], color=[CINZA, VERDE], width=0.8)
for i, v in enumerate([a, b_]): ax.text(i, v + 0.8, br(v, 1) + " km", ha="center", fontweight="bold", fontsize=14, color=TINTA)
ax.annotate(f"−{br(q, 2)}%", xy=(1, b_ + 5), xytext=(0.45, a * 1.2), color=LARANJA, fontweight="bold", fontsize=14,
            arrowprops={"arrowstyle": "-", "color": LARANJA, "lw": 1.5})
ax.set_ylim(0, a * 1.3); ax.yaxis.set_major_formatter(fmt(0)); ax.set_ylabel("Distância média por equipe (km)")
ax.set_title("Distância média por equipe antes e depois\ndo refinamento 2OPT", fontweight="bold")
fig.tight_layout(); fig.savefig("figura4_2opt.png", dpi=200); plt.close(fig)
print("2opt", len(d), a, b_, q)

# ---- Figura 5: rotas ACO + 2OPT em 01/01/2019
s = pd.read_excel(T + "sequencia_atendimentos.xlsx"); s = s[s.dia.astype(str) == "2019-01-01"]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
cmap = plt.get_cmap("tab10")
for e, g in s.groupby("equipe"):
    g = g.sort_values("ordem_atendimento")
    ax.plot(g.lon, g.lat, "-", color=cmap(e % 10), lw=1.6, zorder=2)
    ax.scatter(g.lon, g.lat, s=14, color=cmap(e % 10), zorder=3, label=f"Equipe {e + 1}")
    ax.scatter(g.lon.iloc[:1], g.lat.iloc[:1], s=70, marker="s", facecolor="white", edgecolor=cmap(e % 10), lw=2, zorder=4)
ax.set_aspect(1 / np.cos(np.radians(40.7)))
ax.xaxis.set_major_formatter(fmt(2)); ax.yaxis.set_major_formatter(fmt(2))
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
h, l = ax.get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([], [], marker="s", color="w", markeredgecolor="#555555", markersize=8, markeredgewidth=2)); l.append("Início da rota")
ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
ax.set_title("Rotas otimizadas (ACO + 2OPT) por equipe – 01/01/2019", fontweight="bold")
fig.savefig("figura5_rotas.png", dpi=200, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
print("equipes 01/01", s.equipe.nunique())
