# ==========================================================
# ANALISE COMPLEMENTAR v11 (TCC) - roda SEPARADAMENTE do pipeline
# ==========================================================
# Duas analises pedidas pelo orientador, que nao exigem rodar de novo o
# pipeline principal:
#
#   PARTE 1 - Efeito do balanceamento sob roteamento otimizado:
#     compara "Sem balanceamento + ACO + 2OPT" com "Balanceamento + ACO +
#     2OPT" (este ultimo ja calculado pela v11). A alocacao sem balanceamento
#     e refeita com a MESMA funcao do script principal (alocar_equipes_fifo),
#     que e deterministica; como verificacao, a distancia FIFO recalculada
#     deve coincidir com a da v11 (coluna "confere_fifo_v11").
#
#   PARTE 2 - Curva de convergencia do ACO (melhor distancia x iteracao):
#     roda o ACO, com os parametros da v11, sobre as rotas balanceadas de uma
#     subamostra de dias e registra a melhor distancia a cada iteracao. A
#     funcao usada e uma copia fiel de `otimizar_rota_aco` que apenas guarda
#     o historico; o script confere que o resultado final e identico ao da
#     funcao original com a mesma semente.
#
# Como usar (na mesma pasta do otimizacao_rotas_aco_2opt.py da v11):
#   1) ajuste CAMINHO_SCRIPT_PRINCIPAL, CAMINHO_BASE_TRATADA e PASTA_V11
#   2) python analise_complementar_v11.py
# O script retoma de onde parou se for interrompido (checkpoint por dia).
# Tempo estimado: 1 a 2 h (parte 1) + 10 a 20 min (parte 2).
# ==========================================================
import ast, math, os, sys, time, json
import numpy as np
import pandas as pd

CAMINHO_SCRIPT_PRINCIPAL = "otimizacao_rotas_aco_2opt.py"
CAMINHO_BASE_TRATADA = r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/base_tratada.xlsx"
PASTA_V11 = "saidas_v11"
PASTA_SAIDA = "saidas_complementares_v11"
EXECUTAR_PARTE_1 = True
EXECUTAR_PARTE_2 = True
N_DIAS_CONVERGENCIA = 24          # mesma subamostra estratificada das sensibilidades
LIMITE_DIAS = None                # ex.: 5 para um teste rapido

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ---------- 1. carrega APENAS definicoes e constantes do script principal
def carregar_definicoes(caminho):
    fonte = open(caminho, encoding="utf-8-sig").read()
    arvore = ast.parse(fonte)
    ns = {"__name__": "definicoes_v11"}
    manter = []
    for no in arvore.body:
        if isinstance(no, (ast.Import, ast.ImportFrom)):
            if any(a.name.split(".")[0] in ("osmnx", "networkx", "geopandas") for a in no.names):
                continue  # nao necessarios aqui
            manter.append(no)
        elif isinstance(no, ast.Try) and all(isinstance(b, (ast.Import, ast.ImportFrom, ast.Assign)) for b in no.body):
            manter.append(no)
        elif isinstance(no, ast.FunctionDef):
            manter.append(no)
        elif isinstance(no, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in no.targets):
            nome = no.targets[0].id
            if nome in ("G", "CAMINHO_DIAS_COM_ERRO", "INICIO_SCRIPT"):
                continue
            manter.append(no)
    modulo = ast.Module(body=manter, type_ignores=[])
    exec(compile(modulo, caminho, "exec"), ns)
    ns["REGISTROS_REPLICACOES_ACO_PIPELINE"] = []
    ns["registrar_erro_dia"] = lambda dia, msg: None
    ns["log_etapa"] = lambda msg: None      # silencia o log detalhado do ACO
    return ns

V = carregar_definicoes(CAMINHO_SCRIPT_PRINCIPAL)
print(f"Definicoes carregadas: ACO {V['ACO_FORMIGAS']} formigas x {V['ACO_ITERACOES']} iteracoes; "
      f"km/grau lon = {V['KM_POR_GRAU_LONGITUDE']:.2f}")

# ---------- 2. base tratada (gravada pela propria v11)
base = pd.read_excel(CAMINHO_BASE_TRATADA)
base = base.loc[:, ~base.columns.astype(str).str.startswith("Unnamed")]
base["INCIDENT_DATETIME"] = pd.to_datetime(base["INCIDENT_DATETIME"])
base["dia"] = base["INCIDENT_DATETIME"].dt.date.astype(str)
dias = sorted(base["dia"].unique())
if LIMITE_DIAS:
    dias = dias[:LIMITE_DIAS]
print(f"Base tratada: {len(base)} pontos, {len(dias)} dias")

def rota_aco_2opt(df_eq, dia, equipe, rotulo):
    coords = list(zip(df_eq["lat"], df_eq["lon"]))
    matriz = V["criar_matriz"](coords)
    if len(coords) <= 2:
        path = list(range(len(coords)))
    else:
        path = V["otimizar_rota_aco"](matriz, semente=V["semente_sensibilidade"](dia, equipe, rotulo, 0))
        path = V["melhorar_rota_2opt"](path, matriz)
    return V["avaliar_caminho"](path, df_eq, coords, matriz)

# ==========================================================
# PARTE 1 - balanceamento sob roteamento otimizado
# ==========================================================
if EXECUTAR_PARTE_1:
    ck = os.path.join(PASTA_SAIDA, "parte1_checkpoint.csv")
    feitos = set(pd.read_csv(ck)["dia"].astype(str)) if os.path.exists(ck) else set()
    t0 = time.perf_counter()
    for k, dia in enumerate(dias, 1):
        if dia in feitos:
            continue
        df_dia = base[base["dia"] == dia].copy()
        if len(df_dia) < 2:
            continue
        equipes = V["alocar_equipes_fifo"](df_dia)
        dist_fifo = dist_aco = exc_aco = 0.0
        inviaveis = 0
        for e, eq in enumerate(equipes):
            df_eq = pd.DataFrame(eq["servicos"]).reset_index(drop=True)
            coords = list(zip(df_eq["lat"], df_eq["lon"]))
            matriz = V["criar_matriz"](coords)
            dist_fifo += V["avaliar_caminho"](V["caminho_fifo"](df_eq), df_eq, coords, matriz)["distancia_total_km"]
            m = rota_aco_2opt(df_eq, dia, e, "sem_balanceamento")
            dist_aco += m["distancia_total_km"]; exc_aco += m["jornada_excedida_horas"]; inviaveis += int(not m["rota_viavel"])
        pd.DataFrame([{"dia": dia, "equipes": len(equipes), "distancia_fifo_km": dist_fifo,
                       "distancia_aco_2opt_km": dist_aco, "jornada_excedida_aco_2opt_h": exc_aco,
                       "rotas_inviaveis_aco_2opt": inviaveis}]).to_csv(ck, mode="a", header=not os.path.exists(ck), index=False)
        if k % 10 == 0:
            print(f"  parte 1: {k}/{len(dias)} dias ({time.perf_counter() - t0:.0f}s)")
    sem = pd.read_csv(ck); sem["dia"] = sem["dia"].astype(str)
    abl = pd.read_excel(os.path.join(PASTA_V11, "tabelas", "ablation_study_diario.xlsx")); abl["dia"] = abl["dia"].astype(str)
    piv = abl.pivot(index="dia", columns="cenario", values="distancia_total_km")
    df = sem.join(piv[["Sem balanceamento + FIFO", "Balanceamento + ACO + 2OPT"]], on="dia", how="inner")
    df["confere_fifo_v11"] = np.isclose(df["distancia_fifo_km"], df["Sem balanceamento + FIFO"], rtol=1e-9)
    df["ganho_balanceamento_sob_aco_pct"] = (1 - df["Balanceamento + ACO + 2OPT"] / df["distancia_aco_2opt_km"]) * 100
    df.to_excel(os.path.join(PASTA_SAIDA, "balanceamento_sob_aco_2opt_diario.xlsx"), index=False)
    from scipy.stats import ttest_rel, wilcoxon
    x, y = df["distancia_aco_2opt_km"], df["Balanceamento + ACO + 2OPT"]
    dif = x - y; n = len(df)
    resumo = {
        "dias": n,
        "fifo_reproduzido_identico_%": float(df["confere_fifo_v11"].mean() * 100),
        "sem_balanc_aco2opt_km_media": float(x.mean()), "sem_balanc_aco2opt_km_ic95": float(1.96 * x.std() / math.sqrt(n)),
        "balanc_aco2opt_km_media": float(y.mean()), "balanc_aco2opt_km_ic95": float(1.96 * y.std() / math.sqrt(n)),
        "ganho_razao_medias_%": float((1 - y.mean() / x.mean()) * 100),
        "ganho_medio_diario_%": float(df["ganho_balanceamento_sob_aco_pct"].mean()),
        "ganho_medio_diario_ic95_%": float(1.96 * df["ganho_balanceamento_sob_aco_pct"].std() / math.sqrt(n)),
        "diferenca_media_km": float(dif.mean()), "diferenca_dp_km": float(dif.std()),
        "d_cohen_pareado": float(dif.mean() / dif.std()),
        "p_t_pareado": float(ttest_rel(x, y).pvalue), "p_wilcoxon": float(wilcoxon(x, y).pvalue),
        "rotas_sem_balanc_aco2opt_acima_8h_%": float(sem["rotas_inviaveis_aco_2opt"].sum() / sem["equipes"].sum() * 100),
        "jornada_excedida_sem_balanc_aco2opt_h_dia": float(sem["jornada_excedida_aco_2opt_h"].mean()),
    }
    pd.Series(resumo).to_frame("valor").to_excel(os.path.join(PASTA_SAIDA, "balanceamento_sob_aco_2opt_resumo.xlsx"))
    print("\nPARTE 1 - resumo"); print(json.dumps(resumo, indent=2, ensure_ascii=False))

# ==========================================================
# PARTE 2 - curva de convergencia do ACO
# ==========================================================
def aco_com_historico(matriz, semente):
    """Copia fiel de `otimizar_rota_aco` (mesmas operacoes e mesma ordem de
    sorteios), acrescentando o historico da melhor distancia por iteracao."""
    formigas, iteracoes = V["ACO_FORMIGAS"], V["ACO_ITERACOES"]
    alpha, beta, evap, q = V["ACO_ALPHA"], V["ACO_BETA"], V["ACO_EVAPORACAO"], V["ACO_Q"]
    rng = np.random.RandomState(semente); n = len(matriz)
    ms = matriz.copy(); ms[ms == 0] = 0.001
    fer = np.ones((n, n)); heur = 1 / ms
    melhor = list(range(n)); melhor_d = V["distancia_rota"](melhor, matriz); hist = []
    for _ in range(iteracoes):
        rotas = []
        for _ in range(formigas):
            path = [0]; vis = {0}; atual = 0
            while len(path) < n:
                nv = [i for i in range(n) if i not in vis]
                pesos = np.array([(fer[atual][i] ** alpha) * (heur[atual][i] ** beta) for i in nv])
                s = pesos.sum()
                prox = nv[0] if s == 0 else rng.choice(nv, p=pesos / s)
                path.append(int(prox)); vis.add(int(prox)); atual = int(prox)
            d = V["distancia_rota"](path, matriz); rotas.append((path, d))
            if d < melhor_d:
                melhor, melhor_d = path, d
        fer *= (1 - evap)
        for path, d in rotas:
            dep = q / max(d, 0.001)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]; fer[a][b] += dep; fer[b][a] += dep
        hist.append(melhor_d)
    return melhor, hist

if EXECUTAR_PARTE_2:
    dias_amostra = V["selecionar_dias_sensibilidade"](base, dias, N_DIAS_CONVERGENCIA)
    linhas = []; conferidas = 0
    for dia in dias_amostra:
        equipes = V["alocar_equipes_balanceado"](base[base["dia"] == dia].copy())
        for e, eq in enumerate(equipes):
            df_eq = pd.DataFrame(eq["servicos"]).reset_index(drop=True)
            if len(df_eq) < 3:
                continue
            matriz = V["criar_matriz"](list(zip(df_eq["lat"], df_eq["lon"])))
            semente = V["semente_sensibilidade"](dia, e, "convergencia", 0)
            path, hist = aco_com_historico(matriz, semente)
            if conferidas < 5:  # confere a copia contra a funcao original
                ref = V["otimizar_rota_aco"](matriz, semente=semente)
                assert abs(V["distancia_rota"](ref, matriz) - hist[-1]) < 1e-9, "copia do ACO divergiu da original"
                conferidas += 1
            for it, d in enumerate(hist, 1):
                linhas.append({"dia": dia, "equipe": e, "pontos": len(df_eq), "iteracao": it,
                               "melhor_km": d, "razao_final": d / hist[-1]})
    cv = pd.DataFrame(linhas)
    cv.to_excel(os.path.join(PASTA_SAIDA, "convergencia_aco.xlsx"), index=False)
    curva = cv.groupby("iteracao")["razao_final"].agg(["mean", lambda s: s.quantile(0.9)]).reset_index()
    curva.columns = ["iteracao", "media", "p90"]
    curva["excesso_medio_%"] = (curva["media"] - 1) * 100
    curva.to_excel(os.path.join(PASTA_SAIDA, "convergencia_aco_resumo.xlsx"), index=False)
    for it in (10, 20, 30, 40, 50, 60):
        r = curva[curva.iteracao == it].iloc[0]
        print(f"  iteracao {it}: distancia media {r['excesso_medio_%']:.2f}% acima da final (P90 {100*(r['p90']-1):.2f}%)")
    print(f"PARTE 2 - {cv[['dia','equipe']].drop_duplicates().shape[0]} rotas; copia do ACO conferida em {conferidas} rotas")

print(f"\nArquivos gerados em {PASTA_SAIDA}/. Envie a pasta inteira para atualizacao do TCC.")
