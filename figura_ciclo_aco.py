"""Figura do ciclo do ACO (Ant System) implementado no TCC."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
CINZA, VERDE, TINTA = "#858c95", "#1f7f7a", "#222222"
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(10, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
W, H, Y = 14.5, 17, 17
caixas = [
    (0.5, "Inicialização", "feromônio\nτ = 1 em todos\nos trechos", False),
    (17.5, "Construção", "30 formigas montam\nrotas a partir do\nponto inicial (Eq. 1)", True),
    (34.5, "Avaliação", "comprimento L de\ncada rota; guarda\na melhor", True),
    (51.5, "Evaporação", "todo o feromônio\ndecai: τ ← (1 − ρ)τ\n(ρ = 0,35)", True),
    (68.5, "Depósito", "cada formiga soma\nQ/L aos trechos\nque usou (Eq. 2)", True),
    (85.0, "Saída", "melhor rota das\n60 iterações\n→ refinamento 2OPT", False),
]
for x, tit, txt, ciclo in caixas:
    ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.3,rounding_size=1.2",
                 facecolor="#e8f3f2" if ciclo else "#f2f2f0", edgecolor=VERDE if ciclo else CINZA, lw=1.4))
    ax.text(x + W / 2, Y + H - 3.2, tit, ha="center", va="center", fontweight="bold", color=TINTA, fontsize=10)
    ax.text(x + W / 2, Y + 6.3, txt, ha="center", va="center", color=TINTA, fontsize=8.3, linespacing=1.2)
seta = {"arrowstyle": "-|>", "color": TINTA, "lw": 1.3}
for x0, x1 in ((15.3, 17.2), (32.3, 34.2), (49.3, 51.2), (66.3, 68.2), (83.3, 84.7)):
    ax.annotate("", xy=(x1, Y + H / 2), xytext=(x0, Y + H / 2), arrowprops=seta)
# retorno do ciclo: do deposito para a construcao
ax.annotate("", xy=(24.75, Y - 0.4), xytext=(75.75, Y - 0.4),
            arrowprops={"arrowstyle": "-|>", "color": VERDE, "lw": 1.6, "connectionstyle": "bar,fraction=-0.12"})
ax.text(50.25, 3.2, "próxima iteração (60 no total)", ha="center", va="center", color=VERDE, fontsize=9, fontweight="bold")
ax.set_title("Ciclo do ACO (Ant System) aplicado à rota de cada equipe", fontweight="bold", fontsize=10.5)
fig.tight_layout(); fig.savefig("figura_ciclo_aco.png", dpi=220, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
