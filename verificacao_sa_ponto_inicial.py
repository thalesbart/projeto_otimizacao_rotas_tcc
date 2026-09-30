"""Verificacao complementar: efeito da liberdade de ponto inicial do Simulated Annealing.

Reconstroi, a partir de sequencia_atendimentos.xlsx (v11), as rotas de cada equipe
na mesma subamostra estratificada de 24 dias usada nas analises de sensibilidade
e roda o Simulated Annealing (mesmos parametros do pipeline: solucao inicial Nearest
Neighbor a partir do ponto 0, 2.000 iteracoes, T0 = 10 km, resfriamento 0,995,
trocas 2-opt aleatorias, criterio de Metropolis) em quatro variantes, com 3 sementes:
  - livre_fimpreso : como no pipeline (inicio pode mudar; o ultimo ponto nunca e revertido)
  - fixo_fimpreso  : inicio e fim fixos
  - fixo_fimlivre  : inicio fixo (como ACO, 2OPT e OR-Tools), fim livre  <- comparacao justa
  - livre_fimlivre : inicio e fim livres
O ponto 0 de cada rota e o primeiro atendimento da rota ACO + 2OPT (ordem_atendimento = 1),
o mesmo ponto inicial usado pelos demais metodos. Distancia: projecao equiretangular (v11).
Uso: python verificacao_sa_ponto_inicial.py <pasta saidas_v11>
"""
import sys, math, zlib, numpy as np, pandas as pd
from scipy import stats
T = f"{sys.argv[1] if len(sys.argv) > 1 else 'saidas_v11'}/tabelas/"
s = pd.read_excel(T + "sequencia_atendimentos.xlsx"); s["dia"] = s.dia.astype(str)
KLAT = 111.0; KLON = 111.0 * math.cos(math.radians(40.7))
def M(c):
    c = np.asarray(c); y = c[:, 0] * KLAT; x = c[:, 1] * KLON
    return np.sqrt((y[:, None] - y) ** 2 + (x[:, None] - x) ** 2)
def dist(p, m): return sum(m[p[i], p[i + 1]] for i in range(len(p) - 1))
def nn(m):
    n = len(m); p = [0]; rest = set(range(1, n))
    while rest:
        j = min(rest, key=lambda k: m[p[-1]][k]); p.append(j); rest.remove(j)
    return p
def sa(m, seed, lo, hi_extra):
    rng = np.random.RandomState(seed); n = len(m); p = nn(m); d = dist(p, m); bd = d; T_ = 10.0
    for _ in range(2000):
        i, j = sorted(rng.choice(np.arange(lo, n + hi_extra), size=2, replace=False))
        if j - i >= 2:
            q = p[:i] + p[i:j][::-1] + p[j:]; dq = dist(q, m); de = dq - d
            if de < 0 or rng.random() < math.exp(-de / max(T_, 1e-9)): p, d = q, dq
            bd = min(bd, d)
        T_ *= 0.995
    return bd
dias = sorted(s.dia.unique()); cnt = s.groupby("dia").size().loc[dias].sort_values(kind="stable")
idx = sorted(set(int(round(i)) for i in np.linspace(0, len(cnt) - 1, 24))); amostra = [cnt.index[i] for i in idx]
variantes = {"livre_fimpreso": (0, 0), "fixo_fimpreso": (1, 0), "fixo_fimlivre": (1, 1), "livre_fimlivre": (0, 1)}
res = []
for d_ in amostra:
    for e, ge in s[s.dia == d_].groupby("equipe"):
        first = ge[ge.ordem_atendimento == 1]; rest = ge[ge.ordem_atendimento != 1].sort_values("linha_base_excel")
        ge = pd.concat([first, rest]).reset_index(drop=True); m = M(list(zip(ge.lat, ge.lon)))
        if len(m) < 3: continue
        aco = dist(list(ge.sort_values("ordem_atendimento").index), m)
        seeds = [int(x) for x in np.random.RandomState(zlib.crc32(f"{d_}|{e}".encode()) % 2**31).randint(0, 2**31 - 1, 3)]
        v = {k: np.mean([sa(m, sd, *cfg) for sd in seeds]) for k, cfg in variantes.items()}
        res.append(dict(dia=d_, equipe=e, pontos=len(m), aco_2opt=aco, **v))
r = pd.DataFrame(res); tot = r.groupby("dia").sum(numeric_only=True)
print("rotas", len(r))
for c in variantes:
    print(f"SA {c:16s} {tot[c].mean():8.2f} km  vantagem s/ ACO+2OPT {100*(1-tot[c].mean()/tot.aco_2opt.mean()):+.2f}%  dias melhor {100*(tot[c]<tot.aco_2opt).mean():.1f}%")
# teste pareado (exploratorio) da variante comparavel: inicio fixo, fim livre
x, y = tot.fixo_fimlivre, tot.aco_2opt; dif = x - y; pct = 100 * (x / y - 1)
w = stats.wilcoxon(x, y, method="exact"); tt = stats.ttest_rel(x, y)
resumo = pd.DataFrame([dict(n_dias=len(tot), diferenca_media_pct=pct.mean(), ic95_pct=1.96 * pct.std(ddof=1) / np.sqrt(len(pct)),
    dias_sa_pior_pct=100 * (dif > 0).mean(), p_wilcoxon_exato=w.pvalue, p_t_pareado=tt.pvalue, d_cohen_pareado=dif.mean() / dif.std(ddof=1))])
print(resumo.T)
with pd.ExcelWriter(T + "verificacao_sa_ponto_inicial.xlsx") as xw:
    r.to_excel(xw, sheet_name="rotas", index=False); resumo.to_excel(xw, sheet_name="teste_fixo_fimlivre", index=False)
