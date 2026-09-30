"""Figuras novas do TCC (revisao do orientador): fluxograma do processo e
comparacao pareada entre metodos de roteamento. Usa apenas saidas da v11."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BASE = sys.argv[1] if len(sys.argv) > 1 else "saidas_v11"
CINZA, VERDE, LARANJA, TINTA = "#858c95", "#1f7f7a", "#d9822b", "#222222"
br = lambda v, d=1: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
plt.rcParams.update({"font.size": 10})

# ---------- Figura: fluxograma
fig, ax = plt.subplots(figsize=(10, 2.35)); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
caixas = [
    (1, "1. Dados", "Base FDNY 2019\n(513.723 registros)\nlimpeza e\ngeorreferenciamento\n→ 60.328 pontos,\n364 dias"),
    (21, "2. Alocação", "Sem balanceamento:\nsequencial (referência)\n\nCom balanceamento:\nk-means capacitado +\nabsorção/consolidação"),
    (41, "3. Roteamento", "Por equipe:\nFIFO, Nearest Neighbor,\nACO, ACO + 2OPT,\nSimulated Annealing,\nOR-Tools"),
    (61, "4. Avaliação", "Distância, jornada,\nindicadores de carga,\nfunção objetivo;\nablação e testes\npareados (Holm)"),
    (81, "5. Sensibilidade", "Pesos da função\nobjetivo, parâmetros,\nconvergência e réplicas\ndo ACO; tolerância\ndo balanceamento"),
]
for x, tit, txt in caixas:
    destaque = tit.startswith(("2.", "3."))
    ax.add_patch(FancyBboxPatch((x, 2), 18, 29, boxstyle="round,pad=0.3,rounding_size=1.5",
                                facecolor="#e8f3f2" if destaque else "#f2f2f0", edgecolor=VERDE if destaque else CINZA, lw=1.4))
    ax.text(x + 9, 28, tit, ha="center", va="center", fontweight="bold", color=TINTA, fontsize=10.5)
    ax.text(x + 9, 15, txt, ha="center", va="center", color=TINTA, fontsize=8.6, linespacing=1.25)
for x in (19.3, 39.3, 59.3, 79.3):
    ax.annotate("", xy=(x + 1.5, 16.5), xytext=(x - 0.1, 16.5), arrowprops={"arrowstyle": "-|>", "color": TINTA, "lw": 1.3})
fig.tight_layout(); fig.savefig("figura_fluxograma.png", dpi=220, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------- Figura: diferenca pareada vs ACO + 2OPT
b = pd.read_excel(f"{BASE}/tabelas/metricas_baselines_rotas.xlsx"); b["dia"] = b.dia.astype(str)
d = b.groupby(["dia", "metodo"]).distancia_total_km.sum().unstack()
ref = d["ACO + 2OPT"]; linhas = []
for m in ["ACO sem 2OPT", "OR-Tools", "Simulated Annealing"]:
    x = (d[m] / ref - 1) * 100
    linhas.append((m, x.mean(), 1.96 * x.std() / np.sqrt(len(x)), len(x), False))
v = pd.read_excel(f"{BASE}/tabelas/verificacao_sa_ponto_inicial.xlsx"); v["dia"] = v.dia.astype(str)
tv = v.groupby("dia")[["aco_2opt", "fixo_fimlivre"]].sum(); x = (tv.fixo_fimlivre / tv.aco_2opt - 1) * 100
linhas.append(("Simulated Annealing\n(mesmo ponto inicial;\n24 dias)", x.mean(), 1.96 * x.std() / np.sqrt(len(x)), len(x), True))
nn = ((d["Nearest Neighbor"] / ref - 1) * 100); fifo = ((d["FIFO"] / ref - 1) * 100)
linhas = sorted(linhas, key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (m, mu, ic, n, oco) in enumerate(linhas):
    cor = LARANJA if oco else (VERDE if mu > 0 else CINZA)
    ax.errorbar(mu, i, xerr=ic, fmt="o", ms=8, color=cor, mfc="white" if oco else cor, capsize=4, lw=1.6)
    ax.text(mu, i + 0.28, f"{'+' if mu > 0 else '−'}{br(abs(mu), 2)}%", ha="center", va="bottom", fontsize=9, color=TINTA)
ax.axvline(0, color=TINTA, lw=1); ax.text(0.03, len(linhas) - 0.35, "ACO + 2OPT", fontsize=9, color=TINTA, ha="left")
ax.set_yticks(range(len(linhas))); ax.set_yticklabels([t[0] for t in linhas])
ax.set_ylim(-0.6, len(linhas) - 0.2)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 1)))
ax.set_xlabel("Diferença diária de distância em relação ao ACO + 2OPT (%, média ± IC95)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Diferença de distância em relação ao ACO + 2OPT (mesmo balanceamento)", fontweight="bold", fontsize=10.5)
fig.tight_layout(); fig.savefig("figura_metodos_pareado.png", dpi=220); plt.close(fig)
for t in linhas: print(t[0].split('\n')[0], br(t[1], 2), '±', br(t[2], 2), 'n', t[3])
print('NN', br(nn.mean(), 1), '±', br(1.96 * nn.std() / np.sqrt(len(nn)), 2), '| FIFO', br(fifo.mean(), 1), '±', br(1.96 * fifo.std() / np.sqrt(len(fifo)), 1))
