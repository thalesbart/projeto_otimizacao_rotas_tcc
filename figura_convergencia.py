"""Figura de convergencia do ACO (saidas_complementares_v11/convergencia_aco_resumo.xlsx).
Uso: python figura_convergencia.py <pasta saidas_complementares_v11>"""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = sys.argv[1] if len(sys.argv) > 1 else "saidas_complementares_v11"
CINZA, VERDE, TINTA = "#858c95", "#1f7f7a", "#222222"
br = lambda v, d=1: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
plt.rcParams.update({"font.size": 10})
c = pd.read_excel(f"{P}/convergencia_aco_resumo.xlsx")
media = (c.media - 1) * 100; p90 = (c.p90 - 1) * 100
fig, ax = plt.subplots(figsize=(8, 2.7))
ax.plot(c.iteracao, p90, color=CINZA, lw=1.4, ls="--", label="Percentil 90 das rotas")
ax.plot(c.iteracao, media, color=VERDE, lw=2.2, label="Média das rotas")
for it in (1, 10, 20, 30):
    v = media[c.iteracao == it].iloc[0]
    ax.plot(it, v, "o", color=VERDE, ms=5)
    ax.annotate(f"{br(v, 1)}%", (it, v), xytext=(5, 9), textcoords="offset points", fontsize=9, color=TINTA)
ax.set_xlim(0, 61); ax.set_ylim(0, 62)
ax.set_xlabel("Iteração do ACO")
ax.set_ylabel("Distância acima da\nmelhor rota final (%)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 0)))
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Convergência do ACO: melhor rota encontrada até cada iteração", fontweight="bold", fontsize=10.5)
fig.tight_layout(); fig.savefig("figura_convergencia_aco.png", dpi=220); plt.close(fig)
