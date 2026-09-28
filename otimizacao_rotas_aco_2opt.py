# ==========================================================
# TCC — OTIMIZAÇÃO DE ROTAS E BALANCEAMENTO DE CARGA DE EQUIPES
# BALANCEAMENTO ESPACIAL + ACO + 2-OPT + HTML INTERATIVO
# Versao do pipeline: v11 (ver HISTORICO DE ALTERACOES abaixo)
# ==========================================================
#
# VISÃO GERAL DO SCRIPT
# ----------------------
# Este script implementa e avalia um modelo de roteamento ESTATICO
# (planejamento diario, sem chegada de novas ocorrencias durante a
# execucao nem reotimizacao online) para equipes de atendimento
# emergencial, combinando três componentes:
#
#   1) BALANCEAMENTO OPERACIONAL: distribui as ocorrências do dia entre
#      as equipes de forma a equilibrar a carga de trabalho (heurística
#      gulosa com etapas de consolidação/absorção de equipes ociosas).
#   2) ACO (Ant Colony Optimization): decide, para cada equipe, em que
#      ORDEM visitar as ocorrências atribuídas a ela, minimizando a
#      distância percorrida.
#   3) 2-opt: refina localmente a rota gerada pelo ACO, desfazendo
#      cruzamentos desnecessários.
#
# O pipeline completo, executado dia a dia sobre dados históricos do
# Corpo de Bombeiros de Nova York (FDNY), é:
#
#   ler shapefile (geometria dos ZIP codes)
#       -> ler base de ocorrências (Excel)
#       -> limpar e padronizar datas/ZIPs
#       -> mesclar coordenadas geográficas por ZIP
#       -> baixar a malha viária real da cidade (OSMnx), usada só para
#          desenhar as rotas no HTML interativo, não para o cálculo de
#          distância otimizado (que usa aproximação planar - ver a
#          função `distancia_km`)
#       -> para cada dia:
#            - alocar equipes com e sem balanceamento
#            - para cada equipe, comparar 6 métodos de roteamento
#              (FIFO, Nearest Neighbor, ACO sem 2-opt, ACO + 2-opt,
#              Simulated Annealing e OR-Tools)
#            - registrar métricas (distância, tempo, função objetivo)
#       -> consolidar tudo em DataFrames, exportar para Excel/gráficos
#       -> rodar testes estatísticos (t pareado e Wilcoxon) comparando
#          os cenários par a par
#       -> gerar um mapa HTML interativo com as rotas otimizadas
#
# HISTORICO DE ALTERACOES - v11 (revisao pos-orientacao)
# ------------------------------------------------------
#   1) `distancia_km` e a projecao do k-means passam a usar a projecao
#      equiretangular (Δlon multiplicado por cos(latitude de
#      referencia)); ate a v10, latitude e longitude eram convertidas
#      pelo mesmo fator de 111 km, superestimando em ~32% os trechos
#      leste-oeste na latitude de Nova York.
#   2) A jornada excedida passa a ser medida sobre as ROTAS avaliadas
#      (tempo de atendimento + deslocamento da rota de cada metodo), e
#      nao mais sobre a estimativa Nearest Neighbor usada na alocacao.
#      A funcao objetivo de cada cenario usa esse valor; a estimativa
#      da alocacao continua exportada, com nome proprio.
#   3) O comparativo com o balanceamento bin-packing roda isolado, em
#      bloco try/except proprio: uma falha nele nao descarta mais o dia
#      inteiro do estudo de ablacao (causa do n = 359 da v10).
#   4) Em caso de erro em um dia, todas as listas globais sao revertidas
#      ao estado anterior ao dia (antes, dados parciais permaneciam em
#      algumas tabelas e nao em outras, gerando n diferentes).
#   5) Correcao de Holm-Bonferroni implementada e exportada
#      (colunas p_holm_t e p_holm_wilcoxon).
#   6) ACO (replicacao oficial) e Simulated Annealing passam a usar
#      semente deterministica por (dia, equipe): a execucao e
#      reprodutivel mesmo com checkpoints gerados em rodadas distintas.
#   7) Tempo de execucao de cada metodo de roteamento registrado por
#      rota (metricas_baselines_rotas.xlsx) e resumo de escalabilidade
#      por faixa de tamanho de rota (escalabilidade_metodos.xlsx).
#   8) Analise de sensibilidade da tolerancia distancia/ociosidade
#      incorporada ao script (sensibilidade_tolerancia_ociosidade.xlsx),
#      com grade estendida alem de 60 km/h ociosa.
#   9) Sensibilidade do ACO exporta o desvio do ACO isolado e do
#      ACO + 2-opt, por rota e agregado.
#  10) Figura de boxplot da carga refeita sem winsorizacao, com os
#      outliers visiveis e a media calculada como na Tabela de
#      balanceamento (media das medias diarias).
#
# ==========================================================
# IMPORTAÇÕES
# ==========================================================
# pandas/numpy: manipulação de dados tabulares e vetorizada
# geopandas: leitura do shapefile (dados geográficos vetoriais)
# osmnx/networkx: download e manipulação do grafo da malha viária real
#   (usado apenas para desenhar as rotas no mapa HTML)
# scipy.stats: testes estatísticos (t pareado e Wilcoxon)
# (a correcao de Holm-Bonferroni e implementada localmente, em
#  `ajustar_p_holm`, para nao exigir o statsmodels)

import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
import json
import warnings
import time
import math
import os
import hashlib
import pickle
from scipy.stats import wilcoxon, ttest_rel

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_DISPONIVEL = True
except ImportError:
    routing_enums_pb2 = None
    pywrapcp = None
    ORTOOLS_DISPONIVEL = False

warnings.filterwarnings("ignore")

# Semente global: fixada uma unica vez, no inicio da execucao. A partir da
# v11, as chamadas estocasticas do pipeline (ACO e Simulated Annealing)
# usam sementes LOCAIS deterministicas por (dia, equipe) - ver
# `semente_sensibilidade` -, de modo que o resultado de cada dia nao
# depende da ordem de processamento nem de checkpoints de rodadas
# anteriores. A semente global permanece apenas como salvaguarda.
SEMENTE_ALEATORIA = 42

np.random.seed(SEMENTE_ALEATORIA)

# ==========================================================
# CONFIGURAÇÕES OSMNX
# ==========================================================

ox.settings.use_cache = True
ox.settings.log_console = False

# ==========================================================
# CAMINHOS
# ==========================================================

# Shapefile: arquivo vetorial (SIG) com a geometria (polígonos) de cada
# ZIP code (código postal) dos EUA; usado para obter a latitude/longitude
# aproximada (centroide) de cada ocorrência a partir do seu ZIP code.
PATH_SHAPE = r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/tl_2023_us_zcta520/tl_2023_us_zcta520.shp"

PATH_BASE = r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/archive/fire-incident-dispatch-data.xlsx"

# Copia da base apos limpeza e georreferenciamento (gravada para auditoria).
PATH_BASE_TRATADA = r"C:/Users/Thales/OneDrive - Manager Engenharia Ltda/Área de Trabalho/MBA - Data Science & Analytics/TCC/Dados/base_tratada.xlsx"

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

# Pasta unica onde TODAS as saidas do script (tabelas, graficos, arquivos
# de sensibilidade, manifesto e checkpoints) sao gravadas - os NOMES dos
# arquivos permanecem os mesmos de sempre (o texto do TCC os referencia),
# so o caminho muda. Ver `caminho_tabela`, `caminho_grafico` e
# `caminho_sensibilidade` abaixo.
# v11: pasta NOVA, para que nenhum checkpoint da v10 (calculado com a
# distancia antiga) seja reaproveitado por engano.
PASTA_SAIDA = "saidas_v11"

# Quando True, ignora qualquer checkpoint de dia ja processado (ver secao
# "CHECKPOINT / RETOMADA" mais adiante) e reprocessa tudo do zero.
REPROCESSAR_TUDO = False

# Quando um inteiro, processa apenas os N primeiros dias (na ordem de
# `dias`), em vez do historico completo - usado para testes rapidos do
# pipeline sem esperar a execucao completa (~364 dias). Deixe em None
# para a execucao oficial/completa.
LIMITE_DIAS_TESTE = None

# Quando uma lista de datas (ex.: ["2019-01-01", "2019-03-12"]), processa
# APENAS esses dias - util para um teste rapido que inclua os dias que
# falhavam na v10 (12/03, 13/06, 25/07, 14/09 e 30/09 de 2019). Tem
# precedencia sobre LIMITE_DIAS_TESTE. Deixe em None na execucao oficial.
DIAS_ESPECIFICOS_TESTE = None

for _subpasta in ("tabelas", "graficos", "sensibilidade", "checkpoint"):
    os.makedirs(
        os.path.join(PASTA_SAIDA, _subpasta),
        exist_ok=True
    )


def caminho_tabela(nome_arquivo):
    """Caminho completo (dentro de PASTA_SAIDA/tabelas) para um arquivo
    de saida tabular (.xlsx) - mantem o nome do arquivo, so muda o
    diretorio, para que a estrutura de saidas fique consolidada em uma
    unica pasta (ver secao 4 das instrucoes de alteracao do script)."""

    return os.path.join(PASTA_SAIDA, "tabelas", nome_arquivo)


def caminho_grafico(nome_arquivo):
    """Caminho completo (dentro de PASTA_SAIDA/graficos) para uma figura
    (.png) exportada pelo script."""

    return os.path.join(PASTA_SAIDA, "graficos", nome_arquivo)


def caminho_sensibilidade(nome_arquivo):
    """Caminho completo (dentro de PASTA_SAIDA/sensibilidade) para os
    arquivos gerados pela analise de sensibilidade dos hiperparametros do
    ACO (`analisar_sensibilidade_aco`)."""

    return os.path.join(PASTA_SAIDA, "sensibilidade", nome_arquivo)


PLACE = "New York City, New York, USA"  # área usada para baixar a malha viária real (OSMnx)

# --- Projecao planar (equiretangular) usada em TODAS as distancias -----
# 1 grau de latitude ~ 111 km em qualquer lugar; 1 grau de longitude
# ~ 111 km * cos(latitude). Na latitude de referencia de Nova York
# (40,7 N), 1 grau de longitude ~ 84 km. Para a extensao de uma cidade,
# fixar a latitude de referencia introduz erro inferior a 1%.
KM_POR_GRAU_LATITUDE = 111.0
LATITUDE_REFERENCIA_GRAUS = 40.7
KM_POR_GRAU_LONGITUDE = KM_POR_GRAU_LATITUDE * math.cos(
    math.radians(LATITUDE_REFERENCIA_GRAUS)
)

VELOCIDADE_MEDIA = 40  # km/h constante, usada para converter distância (km) em tempo de deslocamento

TEMPO_BUFFER_MINUTOS = 45  # margem de contingencia por atendimento; NAO entra na restricao de jornada, apenas no indicador tempo_total_com_buffer_horas

JORNADA_MAXIMA_HORAS = 8  # limite legal/contratual de horas trabalhadas por equipe/dia

JORNADA_MAXIMA_SEGUNDOS = (
    JORNADA_MAXIMA_HORAS * 3600
)

# Jornada "alvo": valor um pouco abaixo do máximo (8h) usado como
# referência pelo balanceamento, para deixar uma folga de segurança e
# evitar que pequenas variações estourem o limite máximo.
JORNADA_ALVO_HORAS = 7.95

JORNADA_ALVO_SEGUNDOS = (
    JORNADA_ALVO_HORAS * 3600
)

# Abaixo deste valor, uma equipe é considerada "subutilizada" (baixa
# alocação) e torna-se candidata a receber mais ocorrências ou a ser
# absorvida/consolidada por outras equipes (ver `absorver_equipes_subutilizadas`).
JORNADA_MINIMA_DESEJADA_HORAS = 7.75

JORNADA_MINIMA_DESEJADA_SEGUNDOS = (
    JORNADA_MINIMA_DESEJADA_HORAS * 3600
)

# Quantos km a mais de deslocamento vale a pena percorrer para eliminar
# uma hora de ociosidade de equipe. Controla o trade-off entre os dois
# objetivos do trabalho (distancia x balanceamento de carga) nas fases
# de absorcao e consolidacao do balanceamento.
TOLERANCIA_KM_POR_HORA_OCIOSA = 60.0

COLUNA_ID_OCORRENCIA = "STARFIRE_INCIDENT_ID"

# --- Hiperparâmetros do ACO (Ant Colony Optimization) ---------------
# Ver a função `otimizar_rota_aco` para a explicação de cada papel.
ACO_FORMIGAS = 30       # quantidade de "formigas" (soluções construídas) por iteração
ACO_ITERACOES = 60      # quantas rodadas de construção + atualização de feromônio são executadas
ACO_ALPHA = 1           # peso do feromônio na regra de transição probabilística
ACO_BETA = 2            # peso da heurística local (1/distância) na regra de transição
ACO_EVAPORACAO = 0.35   # fração do feromônio que "evapora" a cada iteração (0 a 1)
ACO_Q = 100             # constante de depósito de feromônio (quanto maior, mais forte o reforço das boas rotas)

# --- Hiperparametros do Simulated Annealing (baseline de roteamento) -
# Ver a funcao `otimizar_rota_simulated_annealing` para o papel de cada um.
SA_TEMPERATURA_INICIAL = 10.0  # temperatura inicial do resfriamento (em unidades de distancia, km)
SA_RESFRIAMENTO = 0.995        # fator de resfriamento geometrico aplicado a cada iteracao (0 a 1)
SA_ITERACOES = 2000            # numero de iteracoes (tentativas de troca 2-opt aleatoria)

# --- Limite de tempo do baseline OR-Tools ---------------------------
ORTOOLS_LIMITE_SEGUNDOS = 2  # tempo maximo (s) de busca local guiada por rota

# --- Replicacoes do ACO no pipeline principal ------------------------
# Quantas vezes o ACO e executado (com sementes distintas e
# deterministicas) para CADA equipe de CADA dia, no pipeline principal
# (nao confundir com a analise de sensibilidade dos hiperparametros,
# que e um estudo a parte). Todas as replicacoes usam um gerador
# aleatorio LOCAL com semente deterministica (via `semente_sensibilidade`,
# a partir de dia, equipe e indice da replicacao). A replicacao de
# indice 0 e a que segue como resultado OFICIAL do pipeline (metricas,
# mapas, ablacao, testes estatisticos); as demais servem para medir a
# dispersao do ACO entre execucoes - ver
# `saidas_v11/tabelas/replicacoes_aco_pipeline.xlsx` e
# `saidas_v11/tabelas/replicacoes_aco_resumo.xlsx`.
N_REPLICACOES_PIPELINE = 5

# --- Pesos da função objetivo (Equação 1 do TCC) --------------------
# F = PESO_DISTANCIA*D + PESO_JORNADA_EXCEDIDA*J + PESO_OCIOSIDADE*O
#     + PESO_DESBALANCEAMENTO*B
# Ver `calcular_funcao_objetivo` e a justificativa de cada peso no
# documento de resultados (Metodologia).
PESO_DISTANCIA = 1.0            # penalidade por km percorrido (termo de referência)
PESO_JORNADA_EXCEDIDA = 3.0     # penalidade mais alta: violar a jornada máxima é mais crítico
PESO_OCIOSIDADE = 0.8           # penalidade mais baixa: ociosidade é parcialmente inevitável
PESO_DESBALANCEAMENTO = 1.5     # penalidade por desequilíbrio de carga entre equipes

NIVEL_CONFIANCA_Z = 1.96  # valor z para IC95% (distribuição normal padrão), usado em `intervalo_confianca_95`


# Marca de tempo do inicio da execucao do script inteiro - usada apenas
# para reportar o tempo total de execucao no MANIFESTO.txt (ver secao 4
# das instrucoes de alteracao: consolidacao de todas as saidas).
INICIO_SCRIPT = time.perf_counter()


def log_etapa(mensagem):
    """Imprime uma mensagem de log com timestamp (HH:MM:SS), usada ao
    longo do script para acompanhar o progresso do processamento
    (etapas de leitura de dados, balanceamento, ACO, 2-opt etc.)."""

    agora = pd.Timestamp.now().strftime("%H:%M:%S")

    print(f"[{agora}] {mensagem}", flush=True)


# Arquivo de erros da EXECUCAO ATUAL: recriado (vazio) a cada execucao do
# script. Na v10 ele era aberto em modo de acrescimo e acumulava as
# mensagens de todas as rodadas anteriores.
CAMINHO_DIAS_COM_ERRO = os.path.join(PASTA_SAIDA, "dias_com_erro.txt")

with open(CAMINHO_DIAS_COM_ERRO, "w", encoding="utf-8") as _f_erro_inicial:
    _f_erro_inicial.write("")


def registrar_erro_dia(dia, mensagem):
    """Acrescenta uma linha `dia: mensagem` ao dias_com_erro.txt da
    execucao atual."""

    with open(CAMINHO_DIAS_COM_ERRO, "a", encoding="utf-8") as _f_erro:
        _f_erro.write(f"{dia}: {mensagem}\n")


# Funil de limpeza da base (Tabela de caracterizacao dos dados, TCC):
# uma linha por etapa de filtragem, com registros antes/removidos/depois.
# Populado por `registrar_etapa_funil` a cada filtro aplicado a `base`.
registros_funil_limpeza = []


def registrar_etapa_funil(nome_etapa, registros_antes, registros_depois):
    """Registra, no funil de limpeza da base, quantos registros entraram
    e saíram de uma etapa de filtragem (dropna, corte de duração, etc.),
    tanto no log quanto na lista `registros_funil_limpeza` (exportada em
    `resultados/caracterizacao_base.xlsx`)."""

    removidos = registros_antes - registros_depois

    log_etapa(
        f"Funil de limpeza - {nome_etapa}: {registros_antes} entraram, "
        f"{removidos} removidos, {registros_depois} restantes"
    )

    registros_funil_limpeza.append({
        "etapa": nome_etapa,
        "registros_antes": registros_antes,
        "registros_removidos": removidos,
        "registros_depois": registros_depois
    })

# ==========================================================
# LEITURA SHAPEFILE
# ==========================================================
# Carrega a geometria de todos os ZIP codes dos EUA e calcula o
# centroide (ponto médio) de cada polígono, que será usado como
# aproximação da latitude/longitude de cada ocorrência (a base original
# só tem o ZIP code, não coordenadas exatas).

print("\nLENDO SHAPEFILE...")

inicio = time.perf_counter()

gdf = gpd.read_file(PATH_SHAPE)

log_etapa(
    f"Shapefile lido: {len(gdf)} registros em "
    f"{time.perf_counter() - inicio:.1f}s"
)

gdf["zip"] = (
    gdf["ZCTA5CE20"]
    .astype(str)
    .str.zfill(5)
)

gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

# ==========================================================
# LEITURA BASE
# ==========================================================
# Carrega o histórico de ocorrências do FDNY (Corpo de Bombeiros de
# Nova York): cada linha é um atendimento, com horário de abertura,
# chegada da equipe ao local, encerramento e ZIP code da ocorrência.

print("\nLENDO BASE...")

inicio = time.perf_counter()

base = pd.read_excel(PATH_BASE)

log_etapa(
    f"Base Excel lida: {len(base)} registros em "
    f"{time.perf_counter() - inicio:.1f}s"
)

base["linha_base_excel"] = base.index + 2

if COLUNA_ID_OCORRENCIA not in base.columns:
    COLUNA_ID_OCORRENCIA = "linha_base_excel"

# ==========================================================
# DATAS
# ==========================================================
# Converte as colunas de data/hora (texto) para o tipo datetime do
# pandas. `errors="coerce"` transforma valores inválidos/ilegíveis em
# NaT (Not a Time) em vez de travar o script - esses registros são
# removidos na etapa de limpeza logo a seguir.

print("\nTRATANDO DATAS...")

base["INCIDENT_DATETIME"] = pd.to_datetime(
    base["INCIDENT_DATETIME"],
    errors="coerce"
)

base["FIRST_ON_SCENE_DATETIME"] = pd.to_datetime(
    base["FIRST_ON_SCENE_DATETIME"],
    errors="coerce"
)

base["INCIDENT_CLOSE_DATETIME"] = pd.to_datetime(
    base["INCIDENT_CLOSE_DATETIME"],
    errors="coerce"
)

# ==========================================================
# LIMPEZA
# ==========================================================
# Remove ocorrências com qualquer uma das três datas ausente/inválida
# (resultado do "coerce" acima) - sem essas datas não é possível
# calcular tempo de serviço nem agrupar por dia.

_registros_antes = len(base)

base = base.dropna(
    subset=[
        "INCIDENT_DATETIME",
        "FIRST_ON_SCENE_DATETIME",
        "INCIDENT_CLOSE_DATETIME"
    ]
)

registrar_etapa_funil(
    "dropna de datas invalidas/ausentes",
    _registros_antes,
    len(base)
)

# ==========================================================
# DIA
# ==========================================================
# Extrai apenas a data (sem hora) do horário de abertura da ocorrência,
# usada para agrupar as ocorrências em simulações diárias (cada dia é
# tratado como uma instância independente do problema de roteamento).

base["dia"] = (
    base["INCIDENT_DATETIME"]
    .dt.date
    .astype(str)
)

# ==========================================================
# TEMPO SERVIÇO
# ==========================================================
# Duração real do atendimento: tempo entre a CHEGADA da equipe ao local
# (FIRST_ON_SCENE_DATETIME) e o ENCERRAMENTO da ocorrência
# (INCIDENT_CLOSE_DATETIME), em segundos. Essa é a parcela de tempo que
# nenhum algoritmo de roteamento pode reduzir - só o deslocamento entre
# atendimentos é otimizável.

base["tempo_servico"] = (

    base["INCIDENT_CLOSE_DATETIME"]

    -

    base["FIRST_ON_SCENE_DATETIME"]

).dt.total_seconds()

# ==========================================================
# LIMPEZA
# ==========================================================
# Remove: (i) registros sem ZIP code ou sem tempo de serviço válido;
# (ii) tempos de serviço não positivos (erro de registro); (iii)
# atendimentos com duração superior a 12h (43200 s), tratados como
# outliers/erros de registro (ex.: ocorrência nunca fechada
# corretamente no sistema).

print("\nLIMPANDO DADOS...")

_registros_antes = len(base)

base = base.dropna(
    subset=[
        "ZIPCODE",
        "tempo_servico"
    ]
)

registrar_etapa_funil(
    "dropna de ZIP/tempo de servico ausentes",
    _registros_antes,
    len(base)
)

_registros_antes = len(base)

base = base[
    base["tempo_servico"] > 0
]

registrar_etapa_funil(
    "filtro de tempo de atendimento valido (> 0s)",
    _registros_antes,
    len(base)
)

_registros_antes = len(base)

base = base[
    base["tempo_servico"] <= 43200
]

registrar_etapa_funil(
    "corte de duracao superior a 12h (outliers)",
    _registros_antes,
    len(base)
)

_registros_antes = len(base)

# Ocorrencias cujo tempo de atendimento sozinho ja excede a jornada
# maxima (JORNADA_MAXIMA_SEGUNDOS, 8h) nunca cabem em nenhuma equipe -
# nem sozinhas - por mais que fiquem abaixo do corte de outlier de 12h
# acima. Remove-las aqui evita que o balanceamento espacial quebre mais
# adiante (_reparar_equipes_sobrecarregadas) com RuntimeError.
base = base[
    base["tempo_servico"] <= JORNADA_MAXIMA_SEGUNDOS
]

registrar_etapa_funil(
    "corte de duracao superior a jornada maxima (inviaveis)",
    _registros_antes,
    len(base)
)

# ==========================================================
# ZIP
# ==========================================================
# Padroniza o ZIP code como texto de 5 dígitos (remove o ".0" que o
# Excel/pandas às vezes adiciona a números lidos como float, e
# preenche com zeros à esquerda quando necessário) para poder cruzar
# com o shapefile.

_registros_antes = len(base)

base["zip"] = (
    base["ZIPCODE"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(5)
)

registrar_etapa_funil(
    "padronizacao do ZIP (5 digitos)",
    _registros_antes,
    len(base)
)

# Mantém apenas 1 registro por combinação (dia, zip): isso reduz o
# volume de pontos processados por dia (o objetivo é simular a
# distribuição geográfica das ocorrências, não replicar cada chamado
# individual no mesmo ZIP).
_registros_antes = len(base)

base = base.drop_duplicates(
    subset=["dia", "zip"],
    keep="first"
)

registrar_etapa_funil(
    "drop_duplicates (dia, zip)",
    _registros_antes,
    len(base)
)

# ==========================================================
# MERGE GEO
# ==========================================================
# Junta (merge) cada ocorrência com a latitude/longitude do centroide
# do seu ZIP code, obtidas do shapefile na etapa anterior.

print("\nFAZENDO MERGE GEO...")

_registros_antes = len(base)

base = base.merge(

    gdf[
        ["zip", "lat", "lon"]
    ],

    on="zip",

    how="left"
)

base = base.dropna(
    subset=["lat", "lon"]
)

registrar_etapa_funil(
    "merge geografico (ZIP sem correspondencia no shapefile)",
    _registros_antes,
    len(base)
)

base.to_excel(PATH_BASE_TRATADA)

print(f"\nREGISTROS GEO VÁLIDOS: {len(base)}")

# ==========================================================
# DIAS
# ==========================================================
# Todos os dias com dados validos (364 dias de 2019, apos a limpeza) -
# amostra sobre a qual todas as medias, desvios padrao, IC95% e testes
# estatisticos do TCC sao calculados. LIMITE_DIAS_TESTE e
# DIAS_ESPECIFICOS_TESTE restringem o processamento apenas em testes.

dias = sorted(
    base["dia"].unique()
)

if DIAS_ESPECIFICOS_TESTE is not None:

    _dias_pedidos = [str(d) for d in DIAS_ESPECIFICOS_TESTE]

    _dias_ausentes = [d for d in _dias_pedidos if d not in dias]

    if _dias_ausentes:
        log_etapa(
            f"DIAS_ESPECIFICOS_TESTE: dias sem dados validos ignorados: "
            f"{_dias_ausentes}"
        )

    dias = [d for d in dias if d in _dias_pedidos]

    log_etapa(
        f"DIAS_ESPECIFICOS_TESTE ativo: processando apenas {dias}"
    )

elif LIMITE_DIAS_TESTE is not None:

    dias = dias[:LIMITE_DIAS_TESTE]

    log_etapa(
        f"LIMITE_DIAS_TESTE ativo: processando apenas os primeiros "
        f"{len(dias)} dia(s) (execucao de teste, nao a base completa)"
    )

log_etapa(
    f"{len(dias)} dias selecionados para processamento "
    f"{'(toda a base)' if (LIMITE_DIAS_TESTE is None and DIAS_ESPECIFICOS_TESTE is None) else '(modo de teste)'}"
)

print("\nDIAS PROCESSADOS:")

for d in dias:
    print(d)

# ==========================================================
# MALHA VIÁRIA
# ==========================================================
# Baixa o grafo real das ruas de Nova York via OSMnx/OpenStreetMap.
# IMPORTANTE: esse grafo é usado apenas para DESENHAR as rotas no mapa
# HTML interativo (função `ox.shortest_path`, mais adiante), seguindo
# ruas de verdade. Ele NÃO é usado para calcular a distância otimizada
# pelo ACO/2-opt/balanceamento - essa distância usa a aproximação
# planar da função `distancia_km` logo abaixo (ver a limitação
# discutida na Metodologia do TCC).

print("\nBAIXANDO MALHA VIÁRIA...")

inicio = time.perf_counter()

G = ox.graph_from_place(

    PLACE,

    network_type="drive",

    simplify=True,

    retain_all=False
)

log_etapa(
    f"Malha carregada: {len(G.nodes)} nodes, {len(G.edges)} arestas em "
    f"{time.perf_counter() - inicio:.1f}s"
)

print("✔ MALHA CARREGADA")

# ==========================================================
# DISTÂNCIA
# ==========================================================
# Aproximação PLANAR (projecao equiretangular): converte a diferenca de
# latitude e de longitude em km e calcula a distancia euclidiana no
# plano. A latitude usa o fator KM_POR_GRAU_LATITUDE (~111 km); a
# longitude usa KM_POR_GRAU_LONGITUDE = 111 km * cos(latitude de
# referencia), ~84 km em Nova York. Ate a v10, a longitude era convertida
# pelo mesmo fator da latitude, o que superestimava em ~32% os trechos
# leste-oeste. A aproximacao continua sem considerar o traçado real das
# ruas (o que uma distancia de rede viaria, ex. OSRM, faria); na escala
# de uma cidade, a curvatura da Terra (tratada por Haversine) tem efeito
# desprezivel. A vantagem e o baixo custo computacional. A limitacao e
# que os valores absolutos de distancia/tempo nao representam o
# deslocamento real das equipes; a comparacao RELATIVA entre metodos
# usa a mesma metrica para todos.

def projetar_km(lat, lon):
    """Projeta coordenadas geograficas (graus) em um plano cartesiano em
    km (projecao equiretangular com latitude de referencia fixa). Aceita
    escalares ou arrays numpy. Usada por `distancia_km`, `criar_matriz`
    e pelo k-means do balanceamento espacial, para que TODAS as etapas
    usem exatamente a mesma metrica."""

    return (
        np.asarray(lat, dtype=float) * KM_POR_GRAU_LATITUDE,
        np.asarray(lon, dtype=float) * KM_POR_GRAU_LONGITUDE
    )


def distancia_km(lat1, lon1, lat2, lon2):

    y1, x1 = projetar_km(lat1, lon1)

    y2, x2 = projetar_km(lat2, lon2)

    return float(np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2))

# ==========================================================
# MATRIZ DISTÂNCIA
# ==========================================================
# Pré-calcula a distância entre TODOS os pares de pontos de uma equipe
# (matriz n x n) uma única vez, para que o ACO e o 2-opt possam
# consultar essas distâncias repetidamente sem recalculá-las. Na v11 o
# calculo e vetorizado (numpy), com resultado identico ao de chamar
# `distancia_km` para cada par.

def criar_matriz(coords):

    n = len(coords)

    if n == 0:
        return np.zeros((0, 0))

    coords_array = np.asarray(coords, dtype=float)

    y, x = projetar_km(coords_array[:, 0], coords_array[:, 1])

    matriz = np.sqrt(
        (y[:, None] - y[None, :]) ** 2
        +
        (x[:, None] - x[None, :]) ** 2
    )

    np.fill_diagonal(matriz, 0.0)

    return matriz

# ==========================================================
# FUNCOES DE ROTA
# ==========================================================

def distancia_rota(path, matriz):
    """Soma as distâncias entre pontos consecutivos de uma rota
    (sequência de índices `path`), usando a matriz de distâncias
    pré-calculada. Essa é a métrica que o ACO e o 2-opt tentam
    minimizar."""

    distancia = 0

    for i in range(len(path) - 1):

        distancia += matriz[
            path[i]
        ][path[i + 1]]

    return distancia


def melhorar_rota_2opt(path, matriz):
    """Refinamento local 2-opt: percorre todos os pares de posições
    (i, j) da rota e testa INVERTER o trecho entre elas (equivalente a
    remover duas arestas que se cruzam e reconectar os pontos na outra
    ordem possível). Se a troca reduzir a distância total, ela é
    mantida; senão, é descartada. O processo se repete até que nenhuma
    troca consiga mais melhorar a rota (ótimo local). É um algoritmo
    guloso e determinístico: sempre parte da rota recebida (aqui, a
    rota construída pelo ACO) e só aceita melhorias, nunca pioras -
    por isso é chamado de "refinamento", e não de busca independente."""

    melhor_path = path.copy()

    melhor_distancia = distancia_rota(
        melhor_path,
        matriz
    )

    melhorou = True

    while melhorou:

        melhorou = False

        for i in range(1, len(melhor_path) - 2):

            for j in range(i + 1, len(melhor_path)):

                if j - i == 1:
                    continue

                # Inverte o trecho [i:j] da rota - essa é a "troca 2-opt"
                # propriamente dita (2 arestas removidas, 2 recriadas).
                novo_path = (
                    melhor_path[:i]
                    +
                    melhor_path[i:j][::-1]
                    +
                    melhor_path[j:]
                )

                nova_distancia = distancia_rota(
                    novo_path,
                    matriz
                )

                if nova_distancia < melhor_distancia:

                    melhor_path = novo_path

                    melhor_distancia = nova_distancia

                    melhorou = True

        if melhorou:
            continue

    return melhor_path

# ==========================================================
# ACO (Ant Colony Optimization)
# ==========================================================
# Metaheurística bioinspirada no comportamento de colônias de formigas
# reais. Cada "formiga" (ver `otimizar_rota_aco` abaixo) constrói uma
# rota completa, ponto a ponto, escolhendo o PRÓXIMO ponto de forma
# PROBABILÍSTICA (não determinística), combinando dois fatores:
#
#   - feromônio[atual][candidato] ** ALPHA: memória coletiva das
#     formigas anteriores - trechos usados em boas rotas acumulam mais
#     feromônio e tendem a ser escolhidos novamente (retroalimentação
#     positiva).
#   - heurística[atual][candidato] ** BETA, onde heurística = 1/distância:
#     preferência "gulosa" por pontos mais próximos.
#
# A cada iteração, todas as formigas constroem uma rota, o feromônio de
# TODOS os trechos evapora um pouco (ACO_EVAPORACAO), e depois é
# reforçado nos trechos usados pelas rotas construídas nesta iteração,
# proporcionalmente a ACO_Q / distância da rota (rotas mais curtas
# depositam mais feromônio). Repetindo esse ciclo por ACO_ITERACOES
# rodadas, o algoritmo tende a convergir para rotas de boa qualidade,
# sem garantir encontrar o ótimo global (por isso o refinamento 2-opt é
# aplicado depois, como uma segunda camada de otimização).

def otimizar_rota_aco(
    matriz,
    formigas=None,
    iteracoes=None,
    alpha=None,
    beta=None,
    evaporacao=None,
    q=None,
    semente=None
):
    """Parametros opcionais (formigas, iteracoes, alpha, beta, evaporacao,
    q): quando omitidos (None), usam as constantes globais ACO_* -
    portanto todas as chamadas existentes continuam produzindo exatamente
    o mesmo resultado de antes. Servem para permitir a analise de
    sensibilidade (`analisar_sensibilidade_aco`), que varia esses valores
    sem tocar no comportamento padrao do pipeline principal.

    `semente`, quando informada, isola a aleatoriedade desta chamada em
    um gerador local (np.random.RandomState), sem afetar o estado global
    de aleatoriedade usado pelo restante do script."""

    formigas = ACO_FORMIGAS if formigas is None else formigas
    iteracoes = ACO_ITERACOES if iteracoes is None else iteracoes
    alpha = ACO_ALPHA if alpha is None else alpha
    beta = ACO_BETA if beta is None else beta
    evaporacao = ACO_EVAPORACAO if evaporacao is None else evaporacao
    q = ACO_Q if q is None else q

    gerador_aleatorio = (
        np.random.RandomState(semente)
        if semente is not None
        else np.random
    )

    n = len(matriz)

    if n <= 2:
        return list(range(n))

    log_etapa(
        f"ACO iniciado: {n} pontos, {formigas} formigas, "
        f"{iteracoes} iteracoes"
    )

    inicio_aco = time.perf_counter()

    matriz_segura = matriz.copy()

    matriz_segura[matriz_segura == 0] = 0.001

    feromonio = np.ones((n, n))

    heuristica = 1 / matriz_segura

    melhor_path = list(range(n))

    melhor_distancia = distancia_rota(
        melhor_path,
        matriz
    )

    for iteracao in range(iteracoes):

        rotas_iteracao = []

        for _ in range(formigas):

            path = [0]

            visitados = set(path)

            atual = 0

            while len(path) < n:

                nao_visitados = [
                    i for i in range(n)
                    if i not in visitados
                ]

                # Regra de transição probabilística do ACO: o "peso" de
                # cada candidato combina feromônio (memória coletiva,
                # elevado a ALPHA) e heurística de proximidade (1/distância,
                # elevada a BETA). Quanto maior o peso, maior a chance de
                # a formiga escolher aquele candidato como próximo destino.
                pesos = np.array([

                    (
                        feromonio[atual][i] ** alpha
                    )
                    *
                    (
                        heuristica[atual][i] ** beta
                    )

                    for i in nao_visitados
                ])

                soma_pesos = pesos.sum()

                if soma_pesos == 0:
                    proximo = nao_visitados[0]
                else:
                    # Normaliza os pesos para somarem 1 (distribuição de
                    # probabilidade) e sorteia o próximo ponto de acordo
                    # com essas probabilidades - é isso que torna o ACO
                    # um algoritmo estocástico, diferente de heurísticas
                    # puramente gulosas como o Nearest Neighbor.
                    probabilidades = pesos / soma_pesos

                    proximo = gerador_aleatorio.choice(
                        nao_visitados,
                        p=probabilidades
                    )

                path.append(int(proximo))

                visitados.add(int(proximo))

                atual = int(proximo)

            distancia = distancia_rota(
                path,
                matriz
            )

            rotas_iteracao.append((
                path,
                distancia
            ))

            if distancia < melhor_distancia:

                melhor_path = path

                melhor_distancia = distancia

        # Evaporação: todo o feromônio da matriz decai por um fator fixo
        # a cada iteração, para que trechos bons de iterações antigas não
        # dominem para sempre e o algoritmo continue explorando.
        feromonio *= (1 - evaporacao)

        # Depósito: cada formiga reforça o feromônio dos trechos que
        # usou, proporcionalmente a ACO_Q / distância da rota - rotas mais
        # curtas depositam mais feromônio, reforçando a tendência de as
        # próximas formigas seguirem por caminhos parecidos com os
        # melhores já encontrados (retroalimentação positiva).
        for path, distancia in rotas_iteracao:

            deposito = q / max(distancia, 0.001)

            for i in range(len(path) - 1):

                a = path[i]
                b = path[i + 1]

                feromonio[a][b] += deposito

                feromonio[b][a] += deposito

        if (
            (iteracao + 1) % 10 == 0
            or
            iteracao + 1 == iteracoes
        ):

            log_etapa(
                f"ACO iteracao {iteracao + 1}/{iteracoes} "
                f"- melhor distancia {melhor_distancia:.2f} km"
            )

    log_etapa(
        f"ACO concluido em {time.perf_counter() - inicio_aco:.1f}s"
    )

    return melhor_path

# ==========================================================
# ACO EXCLUSIVO
# ==========================================================
# Função de conveniência que encadeia os dois componentes de
# roteamento avaliados no estudo de ablação: primeiro o ACO constrói
# uma rota, depois o 2-opt a refina localmente. É este par
# (ACO + 2-opt) que representa a configuração completa do modelo
# proposto no TCC.

def otimizar_rota_aco_exclusivo(coords):

    n = len(coords)

    if n <= 2:
        return list(range(n)), "direta"

    log_etapa(f"Criando matriz de distancia para {n} pontos")

    inicio = time.perf_counter()

    matriz = criar_matriz(coords)

    log_etapa(
        f"Matriz criada em {time.perf_counter() - inicio:.1f}s"
    )

    path_aco = otimizar_rota_aco(matriz)

    log_etapa("Aplicando 2OPT")

    inicio = time.perf_counter()

    path_aco = melhorar_rota_2opt(
        path_aco,
        matriz
    )

    log_etapa(
        f"2OPT concluido em {time.perf_counter() - inicio:.1f}s"
    )

    return path_aco, "ACO+2OPT"


def caminho_fifo(df_eq):
    """Baseline FIFO (First In, First Out): atende as ocorrências na
    ordem cronológica em que foram abertas, ignorando completamente a
    proximidade geográfica entre elas. Representa a ausência de
    qualquer critério de roteamento - é o cenário de referência (pior
    caso) usado para medir o ganho percentual dos demais métodos."""

    return list(
        df_eq.sort_values("INCIDENT_DATETIME").index
    )


def caminho_nearest_neighbor(matriz, inicio=0):
    """Baseline Nearest Neighbor: heurística gulosa clássica para o
    Problema do Caixeiro Viajante - a cada passo, vai para o ponto
    não visitado mais PRÓXIMO do ponto atual. É simples e rápida, mas
    pode ficar "presa" em decisões locais ruins (ex.: deixar um ponto
    isolado para o final, exigindo um deslocamento longo); por isso
    tende a produzir rotas piores que o ACO."""

    n = len(matriz)

    if n <= 2:
        return list(range(n))

    nao_visitados = set(range(n))

    atual = inicio

    path = [atual]

    nao_visitados.remove(atual)

    while len(nao_visitados) > 0:

        proximo = min(
            nao_visitados,
            key=lambda indice: matriz[atual][indice]
        )

        path.append(proximo)

        nao_visitados.remove(proximo)

        atual = proximo

    return path


def otimizar_rota_simulated_annealing(
    matriz,
    temperatura_inicial=None,
    resfriamento=None,
    iteracoes=None,
    semente=None
):
    """Baseline Simulated Annealing: metaheuristica de busca local que,
    diferente do 2-opt (que so aceita melhorias), tambem aceita PIORAS
    com uma probabilidade que diminui ao longo da execucao (a
    "temperatura" esfria geometricamente a cada iteracao), permitindo
    escapar de otimos locais. Solucao inicial: rota do Nearest
    Neighbor. Vizinhanca: troca 2-opt aleatoria (inverte um trecho
    aleatorio da rota). Criterio de aceitacao: Metropolis padrao -
    sempre aceita se a nova rota for melhor; se for pior, aceita com
    probabilidade exp(-delta / temperatura).

    `semente`, quando informada, isola a aleatoriedade desta chamada em
    um gerador local (np.random.RandomState), no mesmo padrao usado
    pelo ACO (`otimizar_rota_aco`)."""

    temperatura_inicial = (
        SA_TEMPERATURA_INICIAL if temperatura_inicial is None
        else temperatura_inicial
    )
    resfriamento = SA_RESFRIAMENTO if resfriamento is None else resfriamento
    iteracoes = SA_ITERACOES if iteracoes is None else iteracoes

    gerador_aleatorio = (
        np.random.RandomState(semente)
        if semente is not None
        else np.random
    )

    n = len(matriz)

    if n <= 2:
        return list(range(n))

    path_atual = caminho_nearest_neighbor(matriz)

    distancia_atual = distancia_rota(path_atual, matriz)

    melhor_path = path_atual.copy()

    melhor_distancia = distancia_atual

    temperatura = temperatura_inicial

    for _ in range(iteracoes):

        # Troca 2-opt aleatoria: sorteia dois pontos de corte e inverte
        # o trecho entre eles - mesma vizinhanca usada por `melhorar_rota_2opt`,
        # mas aqui a escolha do trecho e aleatoria, nao exaustiva.
        i, j = sorted(
            gerador_aleatorio.choice(n, size=2, replace=False)
        )

        if j - i < 2:
            continue

        path_candidato = (
            path_atual[:i]
            +
            path_atual[i:j][::-1]
            +
            path_atual[j:]
        )

        distancia_candidata = distancia_rota(path_candidato, matriz)

        delta = distancia_candidata - distancia_atual

        if delta < 0:
            aceita = True
        else:
            probabilidade_aceitacao = math.exp(
                -delta / max(temperatura, 1e-9)
            )
            aceita = gerador_aleatorio.random() < probabilidade_aceitacao

        if aceita:

            path_atual = path_candidato

            distancia_atual = distancia_candidata

            if distancia_atual < melhor_distancia:

                melhor_path = path_atual.copy()

                melhor_distancia = distancia_atual

        temperatura *= resfriamento

    return melhor_path


def otimizar_rota_ortools(matriz, limite_segundos=None):
    """Baseline OR-Tools: resolve o mesmo problema (encontrar a ordem de
    visita que minimiza a distancia total, partindo do ponto 0) como um
    TSP de veiculo unico usando o solver de roteamento do Google
    OR-Tools. Primeira solucao construida pela heuristica
    PATH_CHEAPEST_ARC, refinada por busca local guiada
    (GUIDED_LOCAL_SEARCH) dentro do limite de tempo informado. O
    OR-Tools exige custos INTEIROS, entao a matriz de distancia (float,
    em km) e convertida multiplicando por 1000 e arredondando (preserva
    3 casas decimais de precisao, equivalente a metros).

    Se a biblioteca `ortools` nao estiver instalada, registra o aviso
    via `log_etapa` e retorna None - o chamador (`avaliar_baselines_rotas`)
    deve tratar esse caso sem quebrar o pipeline."""

    if not ORTOOLS_DISPONIVEL:
        log_etapa(
            "OR-Tools nao disponivel (biblioteca 'ortools' nao instalada) "
            "- baseline OR-Tools ignorado"
        )
        return None

    limite_segundos = (
        ORTOOLS_LIMITE_SEGUNDOS if limite_segundos is None
        else limite_segundos
    )

    n = len(matriz)

    if n <= 2:
        return list(range(n))

    matriz_inteira = np.round(matriz * 1000).astype(int)

    # No sentinela (indice n) com custo zero de/para qualquer no: transforma
    # o ciclo fechado que o OR-Tools resolve por padrao em um CAMINHO ABERTO,
    # que e o que `distancia_rota` mede.
    tamanho = n + 1

    matriz_aberta = np.zeros((tamanho, tamanho), dtype=int)

    matriz_aberta[:n, :n] = matriz_inteira

    gerenciador = pywrapcp.RoutingIndexManager(tamanho, 1, [0], [n])

    roteador = pywrapcp.RoutingModel(gerenciador)

    def callback_distancia(indice_origem, indice_destino):

        no_origem = gerenciador.IndexToNode(indice_origem)

        no_destino = gerenciador.IndexToNode(indice_destino)

        return int(matriz_aberta[no_origem][no_destino])

    indice_callback = roteador.RegisterTransitCallback(callback_distancia)

    roteador.SetArcCostEvaluatorOfAllVehicles(indice_callback)

    parametros_busca = pywrapcp.DefaultRoutingSearchParameters()

    parametros_busca.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    parametros_busca.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    parametros_busca.time_limit.FromSeconds(limite_segundos)

    solucao = roteador.SolveWithParameters(parametros_busca)

    if solucao is None:
        log_etapa("OR-Tools nao encontrou solucao - usando Nearest Neighbor como fallback")
        return caminho_nearest_neighbor(matriz)

    path = []

    indice = roteador.Start(0)

    while not roteador.IsEnd(indice):

        no = gerenciador.IndexToNode(indice)

        if no < n:
            path.append(no)

        indice = solucao.Value(roteador.NextVar(indice))

    return path


def avaliar_caminho(path, df_eq, coords, matriz):
    """Calcula as métricas operacionais de uma rota já definida
    (distância total, tempo de deslocamento, tempo de atendimento,
    tempo total da jornada e se a rota é viável dentro da jornada
    máxima). Usada para avaliar qualquer um dos métodos de roteamento
    de forma padronizada. O tempo total considera a rota efetivamente
    avaliada, de modo que `jornada_excedida_horas` e `rota_viavel`
    refletem a execucao daquela rota (e nao a estimativa usada na
    alocacao)."""

    distancia_total = distancia_rota(
        path,
        matriz
    )

    tempo_deslocamento = (
        distancia_total / VELOCIDADE_MEDIA
    ) * 3600

    tempo_servicos = df_eq["tempo_servico"].sum()

    tempo_total = tempo_servicos + tempo_deslocamento

    return {
        "distancia_total_km": distancia_total,
        "tempo_deslocamento": tempo_deslocamento,
        "tempo_servicos": tempo_servicos,
        "tempo_total": tempo_total,
        "jornada_horas": tempo_total / 3600,
        # Horas acima da jornada maxima NA ROTA AVALIADA (v11) - e este o
        # valor usado no termo J da funcao objetivo.
        "jornada_excedida_horas": max(
            0.0,
            (tempo_total - JORNADA_MAXIMA_SEGUNDOS) / 3600
        ),
        "rota_viavel": tempo_total <= JORNADA_MAXIMA_SEGUNDOS
    }


def calcular_funcao_objetivo(
    distancia_km_total,
    jornada_excedida_horas,
    ociosidade_horas,
    desbalanceamento_horas
):
    """Função objetivo do modelo (Equação 1 do TCC): combina quatro
    critérios concorrentes em um único valor numérico a ser minimizado,
    por meio de uma soma ponderada (técnica comum em otimização
    multiobjetivo). Quanto menor o valor retornado, melhor a solução.
    Os pesos (PESO_*, definidos no topo do script) refletem a
    prioridade relativa de cada critério - ver a justificativa
    detalhada de cada peso no documento de resultados (Metodologia)."""

    return (
        PESO_DISTANCIA * distancia_km_total
        +
        PESO_JORNADA_EXCEDIDA * jornada_excedida_horas
        +
        PESO_OCIOSIDADE * ociosidade_horas
        +
        PESO_DESBALANCEAMENTO * desbalanceamento_horas
    )


def avaliar_baselines_rotas(df_eq, coords, dia=None, equipe_id=None):
    """Para uma mesma equipe (mesmo conjunto de ocorrências), gera e
    avalia as rotas pelos métodos comparados: FIFO, Nearest Neighbor,
    ACO (sem 2-opt), ACO + 2-opt, Simulated Annealing e OR-Tools (quando
    disponível). Isso permite comparar os métodos em igualdade de
    condições - mesmas ocorrências, mesma equipe, mesmo dia - isolando o
    efeito do método de roteamento em si.

    v11:
      - o ACO (todas as replicacoes, inclusive a oficial, de indice 0) e
        o Simulated Annealing usam sementes deterministicas derivadas de
        (dia, equipe), de modo que o resultado nao depende da ordem de
        processamento dos dias nem de checkpoints de rodadas anteriores;
      - o tempo de execucao de cada metodo e medido e devolvido em
        `metricas["tempo_execucao_segundos"]`. Para o ACO + 2-opt, o
        tempo e a soma do ACO com o 2-opt da replicacao oficial.

    Sem `dia`/`equipe_id` (chamadas fora do pipeline principal), roda uma
    unica replicacao, com semente derivada apenas do conteudo da rota."""

    identificar_replicacoes = dia is not None and equipe_id is not None

    chave_dia = dia if dia is not None else "sem-dia"
    chave_equipe = equipe_id if equipe_id is not None else "sem-equipe"

    tempos_execucao = {}

    inicio_metodo = time.perf_counter()
    matriz = criar_matriz(coords)
    tempo_matriz = time.perf_counter() - inicio_metodo

    inicio_metodo = time.perf_counter()
    path_fifo = caminho_fifo(df_eq)
    tempos_execucao["FIFO"] = time.perf_counter() - inicio_metodo

    inicio_metodo = time.perf_counter()
    path_nn = caminho_nearest_neighbor(matriz)
    tempos_execucao["Nearest Neighbor"] = time.perf_counter() - inicio_metodo

    inicio_metodo = time.perf_counter()
    path_sa = otimizar_rota_simulated_annealing(
        matriz,
        semente=semente_sensibilidade(chave_dia, chave_equipe, "sa", 0)
    )
    tempos_execucao["Simulated Annealing"] = time.perf_counter() - inicio_metodo

    inicio_metodo = time.perf_counter()
    path_ortools = otimizar_rota_ortools(matriz)
    tempos_execucao["OR-Tools"] = time.perf_counter() - inicio_metodo

    n_replicacoes = N_REPLICACOES_PIPELINE if identificar_replicacoes else 1

    path_aco = None
    path_aco_2opt = None

    for replicacao in range(n_replicacoes):

        semente_replicacao = semente_sensibilidade(
            chave_dia,
            chave_equipe,
            "pipeline",
            replicacao
        )

        inicio_replicacao = time.perf_counter()

        path_aco_replicacao = otimizar_rota_aco(
            matriz,
            semente=semente_replicacao
        )

        tempo_aco_replicacao = time.perf_counter() - inicio_replicacao

        inicio_2opt = time.perf_counter()

        path_2opt_replicacao = melhorar_rota_2opt(
            path_aco_replicacao,
            matriz
        )

        tempo_2opt_replicacao = time.perf_counter() - inicio_2opt

        if replicacao == 0:
            path_aco = path_aco_replicacao
            path_aco_2opt = path_2opt_replicacao
            tempos_execucao["ACO sem 2OPT"] = tempo_aco_replicacao
            tempos_execucao["ACO + 2OPT"] = (
                tempo_aco_replicacao + tempo_2opt_replicacao
            )

        if identificar_replicacoes:
            REGISTROS_REPLICACOES_ACO_PIPELINE.append({
                "dia": dia,
                "equipe": equipe_id,
                "replicacao": replicacao,
                "quantidade_pontos": len(matriz),
                "distancia_aco_km": distancia_rota(
                    path_aco_replicacao,
                    matriz
                ),
                "distancia_aco_2opt_km": distancia_rota(
                    path_2opt_replicacao,
                    matriz
                ),
                "tempo_execucao_segundos": tempo_aco_replicacao,
                "tempo_execucao_2opt_segundos": tempo_2opt_replicacao
            })

    baselines = {
        "FIFO": path_fifo,
        "Nearest Neighbor": path_nn,
        "ACO sem 2OPT": path_aco,
        "ACO + 2OPT": path_aco_2opt,
        "Simulated Annealing": path_sa
    }

    if path_ortools is not None:
        baselines["OR-Tools"] = path_ortools

    resultados = {}

    for metodo, path in baselines.items():
        metricas_metodo = avaliar_caminho(
            path,
            df_eq,
            coords,
            matriz
        )
        metricas_metodo["tempo_execucao_segundos"] = (
            tempos_execucao.get(metodo, np.nan) + tempo_matriz
        )
        resultados[metodo] = {
            "path": path,
            "metricas": metricas_metodo
        }

    return resultados, matriz

# ==========================================================
# BALANCEAMENTO OPERACIONAL
# ==========================================================
# Conjunto de funções responsável por distribuir as ocorrências do dia
# entre as equipes de forma mais equilibrada, ANTES de qualquer
# roteamento (ACO/2-opt) ser aplicado. A estratégia geral (função
# `alocar_equipes_balanceado`, mais abaixo) é gulosa e em três fases:
#
#   1) alocação inicial: cada ocorrência é atribuída à equipe cujo
#      tempo total ficaria mais próximo da jornada-alvo (7,95h), sem
#      estourar a jornada máxima (8h);
#   2) absorção: equipes muito subutilizadas (< 7,75h) tentam
#      transferir TODAS as suas ocorrências para outras equipes e
#      deixar de existir, reduzindo o número de equipes usadas;
#   3) consolidação: uma varredura adicional tenta mover ocorrências
#      individuais de equipes subutilizadas para outras equipes com
#      mais folga, reduzindo ainda mais o desequilíbrio de carga.
#
# Note que este balanceamento otimiza a DISTRIBUIÇÃO da carga entre
# equipes (quem atende o quê), não a ORDEM de atendimento dentro de
# cada equipe - essa segunda parte é papel do ACO/2-opt.

def calcular_tempo_total_equipe(servicos):

    if len(servicos) == 0:
        return 0

    servicos = sorted(
        servicos,
        key=lambda servico: servico["INCIDENT_DATETIME"]
    )

    tempo_total = 0

    for i, servico in enumerate(servicos):

        tempo_total += servico["tempo_servico"]

        if i > 0:

            anterior = servicos[i - 1]

            dist = distancia_km(

                anterior["lat"],
                anterior["lon"],

                servico["lat"],
                servico["lon"]
            )

            tempo_total += (

                dist / VELOCIDADE_MEDIA

            ) * 3600

    return tempo_total


def estimar_tempo_rota_balanceamento(servicos):
    """Estima rapidamente o tempo total (deslocamento + atendimento)
    que uma equipe levaria para cumprir um conjunto de serviços,
    usando uma heurística Nearest Neighbor simplificada (sempre vai ao
    ponto não visitado mais próximo). Essa estimativa é usada só
    durante o BALANCEAMENTO, para decidir rapidamente se uma
    ocorrência cabe ou não na jornada de uma equipe - por isso usa uma
    heurística mais simples/rápida que o ACO, que só é aplicado depois,
    já com as equipes definidas, para encontrar a melhor ORDEM de
    atendimento."""

    if len(servicos) == 0:
        return 0

    tempo_servicos = sum(
        servico["tempo_servico"] for servico in servicos
    )

    if len(servicos) == 1:
        return tempo_servicos

    nao_visitados = set(
        range(len(servicos))
    )

    atual = min(
        nao_visitados,
        key=lambda indice: servicos[indice]["INCIDENT_DATETIME"]
    )

    nao_visitados.remove(atual)

    distancia_total = 0

    while len(nao_visitados) > 0:

        proximo = min(
            nao_visitados,
            key=lambda indice: distancia_km(
                servicos[atual]["lat"],
                servicos[atual]["lon"],
                servicos[indice]["lat"],
                servicos[indice]["lon"]
            )
        )

        distancia_total += distancia_km(
            servicos[atual]["lat"],
            servicos[atual]["lon"],
            servicos[proximo]["lat"],
            servicos[proximo]["lon"]
        )

        atual = proximo

        nao_visitados.remove(atual)

    tempo_deslocamento = (
        distancia_total / VELOCIDADE_MEDIA
    ) * 3600

    return tempo_servicos + tempo_deslocamento


def estimar_distancia_rota_balanceamento(servicos):
    """Mesma heuristica Nearest Neighbor de `estimar_tempo_rota_balanceamento`,
    mas retorna apenas a distancia estimada (km) da rota, sem converter para
    tempo. Usada pelas fases de pos-processamento (absorcao/consolidacao) do
    balanceamento para impedir que um movimento de ocorrencia entre equipes
    aumente a distancia total percorrida, mesmo quando a jornada continua
    viavel - sem essa checagem, essas fases desfariam o ganho geografico da
    particao espacial (ver `alocar_equipes_balanceado`)."""

    if len(servicos) <= 1:
        return 0

    nao_visitados = set(
        range(len(servicos))
    )

    atual = min(
        nao_visitados,
        key=lambda indice: servicos[indice]["INCIDENT_DATETIME"]
    )

    nao_visitados.remove(atual)

    distancia_total = 0

    while len(nao_visitados) > 0:

        proximo = min(
            nao_visitados,
            key=lambda indice: distancia_km(
                servicos[atual]["lat"],
                servicos[atual]["lon"],
                servicos[indice]["lat"],
                servicos[indice]["lon"]
            )
        )

        distancia_total += distancia_km(
            servicos[atual]["lat"],
            servicos[atual]["lon"],
            servicos[proximo]["lat"],
            servicos[proximo]["lon"]
        )

        atual = proximo

        nao_visitados.remove(atual)

    return distancia_total


def equipe_consegue_atender(servicos):
    """Verifica se um conjunto de serviços cabe dentro da jornada
    máxima (8h) de uma única equipe, usando a estimativa de tempo da
    função acima. É a checagem de viabilidade usada em toda decisão de
    alocação/realocação de ocorrências entre equipes."""

    if len(servicos) == 0:
        return True

    servicos = sorted(
        servicos,
        key=lambda servico: servico["INCIDENT_DATETIME"]
    )

    tempo_total = estimar_tempo_rota_balanceamento(servicos)

    return tempo_total <= JORNADA_MAXIMA_SEGUNDOS


def atualizar_estado_equipe(equipe):

    equipe["servicos"] = sorted(
        equipe["servicos"],
        key=lambda servico: servico["INCIDENT_DATETIME"]
    )

    equipe["tempo_total"] = estimar_tempo_rota_balanceamento(
        equipe["servicos"]
    )

    ultimo = equipe["servicos"][-1]

    equipe["disponivel_em"] = (

        ultimo["INCIDENT_DATETIME"]

        +

        pd.to_timedelta(
            ultimo["tempo_servico"],
            unit="s"
        )
    )


def absorver_equipes_subutilizadas(equipes):
    """Fase de ABSORÇÃO do balanceamento: tenta eliminar equipes cuja
    carga total está abaixo da jornada mínima desejada (7,75h),
    redistribuindo TODOS os seus serviços entre as demais equipes (do
    maior para o menor tempo de serviço, priorizando encaixar primeiro
    os serviços "mais difíceis" de alocar). Se conseguir mover todos os
    serviços de uma equipe sem violar a jornada máxima de nenhuma
    outra, essa equipe é esvaziada e deixa de ser contabilizada -
    reduzindo o número total de equipes utilizadas e, consequentemente,
    aumentando a carga média das equipes remanescentes."""

    equipes = [
        equipe for equipe in equipes
        if len(equipe["servicos"]) > 0
    ]

    absorvidas = 0

    houve_absorcao = True

    while houve_absorcao:

        houve_absorcao = False

        for indice_origem, equipe_origem in sorted(
            list(enumerate(equipes)),
            key=lambda item: item[1]["tempo_total"]
        ):

            if equipe_origem["tempo_total"] >= JORNADA_MINIMA_DESEJADA_SEGUNDOS:
                continue

            if len(equipes) <= 1:
                break

            estado_teste = {
                indice: list(equipe["servicos"])
                for indice, equipe in enumerate(equipes)
                if indice != indice_origem
            }

            servicos_origem = sorted(
                equipe_origem["servicos"],
                key=lambda servico: servico["tempo_servico"],
                reverse=True
            )

            plano_viavel = True

            # Distancia de referencia (geografica) que cada servico "trazia"
            # para a equipe de origem: usada abaixo para impedir que a
            # absorcao aumente a distancia total do sistema, mesmo que a
            # jornada continue viavel (sem essa guarda, a absorcao ignora
            # geografia e desfaz o ganho da particao espacial).
            distancia_origem_completa = estimar_distancia_rota_balanceamento(
                equipe_origem["servicos"]
            )

            for servico in servicos_origem:

                servicos_origem_sem_este = [
                    outro for outro in equipe_origem["servicos"]
                    if outro is not servico
                ]

                tempo_origem_sem_este = estimar_tempo_rota_balanceamento(
                    servicos_origem_sem_este
                )

                marginal_origem = (
                    distancia_origem_completa
                    -
                    estimar_distancia_rota_balanceamento(
                        servicos_origem_sem_este
                    )
                )

                candidatos = []

                for indice_destino, servicos_destino in estado_teste.items():

                    servicos_teste = (
                        servicos_destino
                        +
                        [servico]
                    )

                    tempo_total_teste = estimar_tempo_rota_balanceamento(
                        servicos_teste
                    )

                    if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                        continue

                    marginal_destino = (
                        estimar_distancia_rota_balanceamento(servicos_teste)
                        -
                        estimar_distancia_rota_balanceamento(servicos_destino)
                    )

                    # Trade-off explicito distancia x ociosidade (em vez de
                    # exigir que a distancia nunca aumente, o que e restritivo
                    # demais quando a particao espacial ja deixa cada cluster
                    # quase otimo - nesse caso quase todo movimento aumentaria
                    # a distancia e a absorcao nunca aconteceria): aceita o
                    # movimento se o km adicional gasto por hora de ociosidade
                    # eliminada ficar dentro de TOLERANCIA_KM_POR_HORA_OCIOSA.
                    delta_distancia_km = marginal_destino - marginal_origem

                    tempo_destino_antes = estimar_tempo_rota_balanceamento(
                        servicos_destino
                    )

                    deficit_antes_par = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            equipe_origem["tempo_total"]
                        )
                        +
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_destino_antes
                        )
                    )

                    deficit_depois_par = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_origem_sem_este
                        )
                        +
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_total_teste
                        )
                    )

                    delta_horas_ociosas_eliminadas = (
                        (deficit_antes_par - deficit_depois_par) / 3600
                    )

                    if (
                        delta_distancia_km
                        >
                        TOLERANCIA_KM_POR_HORA_OCIOSA
                        * delta_horas_ociosas_eliminadas
                    ):
                        continue

                    if len(servicos_destino) > 0:

                        ultimo = servicos_destino[-1]

                        dist = distancia_km(
                            ultimo["lat"],
                            ultimo["lon"],
                            servico["lat"],
                            servico["lon"]
                        )

                    else:

                        dist = 0

                    score = (
                        max(
                            0,
                            JORNADA_ALVO_SEGUNDOS - tempo_total_teste
                        ),
                        abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                        dist,
                        len(servicos_destino)
                    )

                    candidatos.append(
                        (
                            score,
                            indice_destino,
                            tempo_total_teste
                        )
                    )

                if len(candidatos) == 0:

                    plano_viavel = False

                    break

                _, melhor_destino, _ = min(
                    candidatos,
                    key=lambda candidato: candidato[0]
                )

                estado_teste[melhor_destino].append(servico)

            if not plano_viavel:
                continue

            for indice_destino, servicos_destino in estado_teste.items():

                equipes[indice_destino]["servicos"] = servicos_destino

                atualizar_estado_equipe(
                    equipes[indice_destino]
                )

            equipe_origem["servicos"] = []
            equipe_origem["tempo_total"] = 0

            absorvidas += 1
            houve_absorcao = True

            equipes = [
                equipe for equipe in equipes
                if len(equipe["servicos"]) > 0
            ]

            break

    return equipes, absorvidas


def consolidar_equipes(equipes):
    """Fase de CONSOLIDAÇÃO do balanceamento: complementar à absorção
    acima, tenta mover serviços INDIVIDUAIS (não necessariamente todos)
    de equipes subutilizadas para outras equipes com mais folga,
    reduzindo o "déficit" total em relação à jornada mínima desejada.
    Diferente da absorção (que só move um serviço se conseguir esvaziar
    a equipe inteira), aqui cada serviço é avaliado e movido
    individualmente, desde que isso reduza o desequilíbrio agregado
    entre a equipe de origem e a de destino."""

    houve_movimento = True
    movimentos = 0
    iteracoes_consolidacao = 0

    # Teto de iteracoes: com TOLERANCIA_KM_POR_HORA_OCIOSA permitindo que a
    # distancia aumente em troca de reduzir ociosidade, o criterio de aceite
    # deixa de ser estritamente monotonico (nao ha mais garantia de que a
    # distancia total so diminui a cada movimento) - por isso duas equipes
    # poderiam, em teoria, trocar o mesmo servico repetidamente (cada troca
    # parecendo vantajosa a partir de uma perspectiva diferente). O teto
    # evita um laco muito longo/infinito nesse cenario.
    while houve_movimento and iteracoes_consolidacao < 200:

        iteracoes_consolidacao += 1
        houve_movimento = False

        equipes = [
            equipe for equipe in equipes
            if len(equipe["servicos"]) > 0
        ]

        equipes = sorted(
            equipes,
            key=lambda equipe: equipe["tempo_total"]
        )

        for equipe_origem in equipes:

            if equipe_origem["tempo_total"] >= JORNADA_MINIMA_DESEJADA_SEGUNDOS:
                continue

            for servico in list(equipe_origem["servicos"]):

                servicos_origem_restantes = [
                    servico_origem for servico_origem in equipe_origem["servicos"]
                    if servico_origem is not servico
                ]

                tempo_origem_restante = estimar_tempo_rota_balanceamento(
                    servicos_origem_restantes
                )

                candidatos = []

                for equipe_destino in equipes:

                    if equipe_destino is equipe_origem:
                        continue

                    servicos_teste = (
                        equipe_destino["servicos"]
                        +
                        [servico]
                    )

                    tempo_total_teste = estimar_tempo_rota_balanceamento(
                        servicos_teste
                    )

                    if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                        continue

                    # Trade-off explicito distancia x ociosidade (ver
                    # TOLERANCIA_KM_POR_HORA_OCIOSA no topo do script): em vez
                    # de exigir que a soma das distancias de origem+destino
                    # nunca aumente (restritivo demais quando a particao
                    # espacial ja deixa cada cluster quase otimo - nesse caso
                    # quase todo movimento aumentaria a distancia e a
                    # consolidacao nunca aconteceria), aceita o movimento se
                    # o km adicional gasto por hora de ociosidade eliminada
                    # ficar dentro da tolerancia.
                    distancia_origem_antes = estimar_distancia_rota_balanceamento(
                        equipe_origem["servicos"]
                    )

                    distancia_destino_antes = estimar_distancia_rota_balanceamento(
                        equipe_destino["servicos"]
                    )

                    distancia_origem_depois = estimar_distancia_rota_balanceamento(
                        servicos_origem_restantes
                    )

                    distancia_destino_depois = estimar_distancia_rota_balanceamento(
                        servicos_teste
                    )

                    delta_distancia_km = (
                        (distancia_origem_depois + distancia_destino_depois)
                        -
                        (distancia_origem_antes + distancia_destino_antes)
                    )

                    deficit_antes = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            equipe_origem["tempo_total"]
                        )
                        +
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            equipe_destino["tempo_total"]
                        )
                    )

                    deficit_depois = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_origem_restante
                        )
                        +
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS
                            -
                            tempo_total_teste
                        )
                    )

                    delta_horas_ociosas_eliminadas = (
                        (deficit_antes - deficit_depois) / 3600
                    )

                    if (
                        len(servicos_origem_restantes) > 0
                        and
                        delta_distancia_km
                        >
                        TOLERANCIA_KM_POR_HORA_OCIOSA
                        * delta_horas_ociosas_eliminadas
                    ):
                        continue

                    dist = distancia_km(

                        equipe_destino["servicos"][-1]["lat"],
                        equipe_destino["servicos"][-1]["lon"],

                        servico["lat"],
                        servico["lon"]
                    )

                    score = (
                        max(
                            0,
                            JORNADA_MINIMA_DESEJADA_SEGUNDOS - tempo_total_teste
                        ),
                        abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                        dist
                    )

                    candidatos.append(
                        (
                            score,
                            equipe_destino
                        )
                    )

                candidatos = sorted(
                    candidatos,
                    key=lambda candidato: candidato[0]
                )

                for _, equipe_destino in candidatos:

                    servicos_teste = (
                        equipe_destino["servicos"]
                        +
                        [servico]
                    )

                    if equipe_consegue_atender(servicos_teste):

                        equipe_destino["servicos"].append(servico)

                        for posicao, servico_origem in enumerate(
                            equipe_origem["servicos"]
                        ):

                            if servico_origem is servico:

                                del equipe_origem["servicos"][posicao]

                                break

                        atualizar_estado_equipe(equipe_destino)

                        if len(equipe_origem["servicos"]) > 0:
                            atualizar_estado_equipe(equipe_origem)
                        else:
                            equipe_origem["tempo_total"] = 0

                        movimentos += 1

                        houve_movimento = True

                        break

                if houve_movimento:
                    break

            if houve_movimento:
                break

    if houve_movimento and iteracoes_consolidacao >= 200:
        log_etapa(
            f"Consolidacao operacional: teto de {iteracoes_consolidacao} "
            f"iteracoes atingido (possivel oscilacao entre equipes) - "
            f"interrompida por seguranca"
        )

    log_etapa(
        f"Consolidacao operacional: {movimentos} ocorrencias realocadas em "
        f"{iteracoes_consolidacao} iteracoes"
    )

    return sorted(
        equipes,
        key=lambda equipe: equipe["tempo_total"],
        reverse=True
    )


def criar_equipe(servico):
    """Cria uma nova equipe contendo apenas o serviço informado (usada
    quando nenhuma equipe existente consegue absorver a ocorrência sem
    violar a jornada máxima)."""

    return {

        "tempo_total": servico["tempo_servico"],

        "disponivel_em": (

            servico["INCIDENT_DATETIME"]

            +

            pd.to_timedelta(
                servico["tempo_servico"],
                unit="s"
            )
        ),

        "servicos": [
            servico
        ]
    }


def alocar_equipes_bin_packing(df):
    """Fase 1 (alocação inicial) do balanceamento operacional: percorre
    as ocorrências do dia da MAIOR para a MENOR duração de atendimento
    (heurística "first-fit decreasing", comum em problemas de
    empacotamento/bin-packing) e atribui cada uma à equipe existente
    cujo tempo total ficaria mais próximo da jornada-alvo (7,95h) sem
    ultrapassar a jornada máxima (8h); se nenhuma equipe existente
    comportar a ocorrência, cria-se uma equipe nova. Processar as
    ocorrências mais longas primeiro reduz o risco de "sobrar" um
    serviço grande sem nenhuma equipe com espaço suficiente no final do
    processo. Depois da alocação inicial, chama as fases de absorção e
    consolidação (acima) para compactar ainda mais o número de equipes."""

    inicio_balanceamento = time.perf_counter()

    servicos = [

        linha for _, linha in df.sort_values(
            by=[
                "tempo_servico",
                "INCIDENT_DATETIME"
            ],
            ascending=[
                False,
                True
            ]
        ).iterrows()
    ]

    total_servicos = len(servicos)

    log_etapa(
        f"Balanceamento iniciado: {total_servicos} ocorrencias pendentes"
    )

    equipes = []

    for indice, servico in enumerate(servicos, start=1):

        melhor_equipe = None
        melhor_score = None

        for equipe in equipes:

            estimativa_minima = (
                equipe["tempo_total"]
                +
                servico["tempo_servico"]
            )

            if estimativa_minima > JORNADA_MAXIMA_SEGUNDOS:
                continue

            servicos_teste = (
                equipe["servicos"]
                +
                [servico]
            )

            tempo_total_teste = estimar_tempo_rota_balanceamento(
                servicos_teste
            )

            if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                continue

            ultimo = equipe["servicos"][-1]

            dist = distancia_km(

                ultimo["lat"],
                ultimo["lon"],

                servico["lat"],
                servico["lon"]
            )

            deficit_jornada = max(
                0,
                JORNADA_ALVO_SEGUNDOS - tempo_total_teste
            )

            folga_jornada = max(
                0,
                JORNADA_MAXIMA_SEGUNDOS - tempo_total_teste
            )

            score = (
                deficit_jornada,
                abs(JORNADA_ALVO_SEGUNDOS - tempo_total_teste),
                folga_jornada,
                dist
            )

            if (
                melhor_score is None
                or
                score < melhor_score
            ):

                melhor_equipe = equipe
                melhor_score = score

        if melhor_equipe is None:

            equipes.append(
                criar_equipe(servico)
            )

        else:

            melhor_equipe["servicos"].append(servico)

            atualizar_estado_equipe(melhor_equipe)

        if (
            indice % 100 == 0
            or
            indice == total_servicos
        ):

            log_etapa(
                f"Balanceamento: {indice}/{total_servicos} ocorrencias, "
                f"{len(equipes)} equipes criadas"
            )

    log_etapa(
        f"Consolidando equipes subutilizadas abaixo de "
        f"{JORNADA_MINIMA_DESEJADA_HORAS:.2f}h"
    )

    equipes, equipes_absorvidas = absorver_equipes_subutilizadas(
        equipes
    )

    log_etapa(
        f"Compactacao inicial: {equipes_absorvidas} equipes absorvidas"
    )

    equipes = consolidar_equipes(equipes)

    equipes, equipes_absorvidas = absorver_equipes_subutilizadas(
        equipes
    )

    log_etapa(
        f"Compactacao final: {equipes_absorvidas} equipes absorvidas"
    )

    log_etapa(
        f"Balanceamento concluido: {len(equipes)} equipes em "
        f"{time.perf_counter() - inicio_balanceamento:.1f}s"
    )

    _validar_balanceamento(df, equipes, rotulo="bin-packing")

    return equipes


def _reparar_equipes_sobrecarregadas(equipes):
    """Rede de seguranca final do balanceamento: se, apos as fases A/B e
    absorcao/consolidacao, alguma equipe AINDA assim excede a jornada
    maxima (pode acontecer no caminho de ultimo recurso de
    `alocar_equipes_balanceado`, quando nenhum k testado produziu uma
    alocacao totalmente viavel e o resultado usado nao passa pela
    validacao de cada tentativa - ou porque a heuristica Nearest Neighbor
    usada para estimar o tempo de rota NAO e estritamente monotonica ao
    remover uma ocorrencia, entao uma unica remocao pode nao ser
    suficiente), remove ocorrencias do fim da equipe sobrecarregada - uma
    de cada vez, ate esvazia-la se preciso - e as realoca para outra
    equipe com folga ou, se nenhuma comportar, para uma equipe nova."""

    equipes = [
        equipe for equipe in equipes
        if len(equipe["servicos"]) > 0
    ]

    for equipe in equipes:
        atualizar_estado_equipe(equipe)

    houve_reparo = False

    indice = 0

    while indice < len(equipes):

        equipe = equipes[indice]

        while (
            len(equipe["servicos"]) > 0
            and
            equipe["tempo_total"] > JORNADA_MAXIMA_SEGUNDOS
        ):

            houve_reparo = True

            servico_removido = equipe["servicos"].pop()

            if len(equipe["servicos"]) > 0:
                atualizar_estado_equipe(equipe)
            else:
                equipe["tempo_total"] = 0

            if (
                servico_removido["tempo_servico"]
                > JORNADA_MAXIMA_SEGUNDOS
            ):
                # Uma unica ocorrencia cujo tempo de atendimento sozinho ja
                # excede a jornada maxima nunca vai caber em equipe nenhuma
                # (nem sozinha) - nao e um problema de balanceamento, e um
                # problema de dado (tempo_servico anormalmente alto para
                # essa ocorrencia). Sinaliza isso explicitamente em vez de
                # entrar em loop tentando (em vao) mover a ocorrencia.
                raise RuntimeError(
                    f"Balanceamento espacial: a ocorrencia "
                    f"{servico_removido[COLUNA_ID_OCORRENCIA]} tem tempo "
                    f"de atendimento sozinha "
                    f"({servico_removido['tempo_servico'] / 3600:.2f}h) "
                    f"maior que a jornada maxima "
                    f"({JORNADA_MAXIMA_HORAS}h) - verifique o dado de "
                    f"tempo_servico dessa ocorrencia"
                )

            destino_encontrado = None

            for outra_equipe in equipes:

                if outra_equipe is equipe:
                    continue

                if equipe_consegue_atender(
                    outra_equipe["servicos"] + [servico_removido]
                ):
                    destino_encontrado = outra_equipe
                    break

            if destino_encontrado is not None:

                destino_encontrado["servicos"].append(servico_removido)

                atualizar_estado_equipe(destino_encontrado)

            else:

                nova_equipe = {"servicos": [servico_removido]}

                atualizar_estado_equipe(nova_equipe)

                equipes.append(nova_equipe)

        indice += 1

    equipes = [
        equipe for equipe in equipes
        if len(equipe["servicos"]) > 0
    ]

    if houve_reparo:
        log_etapa(
            "Reparo de sobrecarga: uma ou mais equipes excediam a jornada "
            "maxima apos o balanceamento - ocorrencias excedentes "
            "realocadas para restaurar a viabilidade"
        )

    return equipes


def _validar_balanceamento(df, equipes, rotulo=""):
    """Confere, ao final de um balanceamento diario, que (i) todas as
    ocorrencias do dia foram alocadas exatamente uma vez (nenhuma perdida,
    nenhuma duplicada) e (ii) nenhuma equipe excede a jornada maxima. Levanta
    excecao com mensagem clara se alguma das duas checagens falhar - o
    pipeline nao deve seguir adiante com um balanceamento inconsistente."""

    ids_esperados = set(df[COLUNA_ID_OCORRENCIA])

    ids_alocados = []

    for equipe in equipes:
        for servico in equipe["servicos"]:
            ids_alocados.append(servico[COLUNA_ID_OCORRENCIA])

    ids_alocados_set = set(ids_alocados)

    if len(ids_alocados) != len(ids_alocados_set):

        duplicados = len(ids_alocados) - len(ids_alocados_set)

        raise RuntimeError(
            f"Balanceamento {rotulo}: {duplicados} ocorrencia(s) "
            f"duplicada(s) entre equipes"
        )

    perdidas = ids_esperados - ids_alocados_set
    extras = ids_alocados_set - ids_esperados

    if len(perdidas) > 0 or len(extras) > 0:
        raise RuntimeError(
            f"Balanceamento {rotulo}: {len(perdidas)} ocorrencia(s) "
            f"perdida(s) e {len(extras)} ocorrencia(s) indevida(s) "
            f"em relacao ao dia processado"
        )

    for indice_equipe, equipe in enumerate(equipes):

        tempo_real = estimar_tempo_rota_balanceamento(equipe["servicos"])

        if tempo_real > JORNADA_MAXIMA_SEGUNDOS:
            raise RuntimeError(
                f"Balanceamento {rotulo}: equipe {indice_equipe} excede a "
                f"jornada maxima ({tempo_real / 3600:.2f}h > "
                f"{JORNADA_MAXIMA_HORAS}h)"
            )

    log_etapa(
        f"Validacao do balanceamento {rotulo}: OK - "
        f"{len(ids_alocados_set)} ocorrencias alocadas em {len(equipes)} "
        f"equipes, nenhuma jornada excedida"
    )


def alocar_equipes_balanceado(df):
    """Balanceamento operacional PRINCIPAL do pipeline: substitui a heuristica
    gulosa sequencial de `alocar_equipes_bin_packing` (que aloca ocorrencia a
    ocorrencia, praticamente cega a geografia) por um algoritmo em DUAS FASES
    que particiona as ocorrencias do dia espacialmente antes de qualquer
    roteamento:

      FASE A (particao espacial com restrição de capacidade, tipo k-means
      capacitado): agrupa as ocorrencias em clusters geograficamente
      compactos, respeitando a jornada maxima de cada cluster/equipe.

      FASE B (refinamento de carga): para equipes ociosas (abaixo da jornada
      minima desejada), tenta mover ocorrencias apenas para equipes
      VIZINHAS geograficamente, e só aceita o movimento se ele nao aumentar
      a distancia total estimada das duas rotas envolvidas.

    O resultado é que cada equipe atende uma area compacta da cidade, o que
    da ao ACO/2-opt (que so decidem a ORDEM de visita, nao o AGRUPAMENTO)
    uma base muito mais favoravel para reduzir a distancia percorrida."""

    inicio_balanceamento = time.perf_counter()

    servicos = [
        linha for _, linha in df.iterrows()
    ]

    total_servicos = len(servicos)

    log_etapa(
        f"Balanceamento espacial iniciado: {total_servicos} ocorrencias "
        f"pendentes"
    )

    if total_servicos == 0:
        return []

    # Projecao planar equiretangular (a MESMA de `distancia_km`, via
    # `projetar_km`): o k-means agrupa as ocorrencias na mesma metrica
    # usada pelo roteamento.
    _lat_km, _lon_km = projetar_km(
        [servico["lat"] for servico in servicos],
        [servico["lon"] for servico in servicos]
    )
    pontos = np.column_stack([_lat_km, _lon_km])

    def _inicializar_centroides_kmeanspp(pontos, k, rng):

        n_pontos = len(pontos)

        indices_centroides = [
            int(rng.randint(0, n_pontos))
        ]

        for _ in range(1, k):

            centroides_atuais = pontos[indices_centroides]

            distancias_quadradas = np.min(
                np.sum(
                    (pontos[:, None, :] - centroides_atuais[None, :, :]) ** 2,
                    axis=2
                ),
                axis=1
            )

            soma_distancias = distancias_quadradas.sum()

            if soma_distancias <= 0:
                proximo_indice = int(rng.randint(0, n_pontos))
            else:
                probabilidades = distancias_quadradas / soma_distancias
                proximo_indice = int(
                    rng.choice(n_pontos, p=probabilidades)
                )

            indices_centroides.append(proximo_indice)

        return pontos[indices_centroides].copy()

    def _centroide_equipe(equipe):

        latitudes = [servico["lat"] for servico in equipe["servicos"]]
        longitudes = [servico["lon"] for servico in equipe["servicos"]]

        _lat_km, _lon_km = projetar_km(
            sum(latitudes) / len(latitudes),
            sum(longitudes) / len(longitudes)
        )
        return (float(_lat_km), float(_lon_km))

    def _executar_fase_a(k):
        """FASE A (particao espacial com restricao de capacidade, tipo
        k-means capacitado) para um `k` especifico. Retorna a lista de
        equipes/clusters e um booleano indicando se foi necessario criar
        algum cluster extra por falta de vaga (sinal de que o `k` testado
        esta mal dimensionado - ver busca do menor k viavel, mais abaixo)."""

        # Gerador de aleatoriedade LOCAL: nao usa `np.random` diretamente,
        # para nao consumir/deslocar o stream global de aleatoriedade que o
        # ACO depende para ser reprodutivel (ver SEMENTE_ALEATORIA no topo).
        # Recriado a cada tentativa de k para que a busca seja determinstica
        # e independente da ordem das tentativas.
        rng_local = np.random.RandomState(SEMENTE_ALEATORIA)

        centroides = _inicializar_centroides_kmeanspp(pontos, k, rng_local)

        clusters_servicos = None
        clusters_indices = None
        houve_cluster_extra = False

        for iteracao in range(30):

            # Distancia (euclidiana, no plano projetado) de cada ocorrencia
            # a cada centroide atual.
            distancias_centroides = np.sqrt(
                np.sum(
                    (pontos[:, None, :] - centroides[None, :, :]) ** 2,
                    axis=2
                )
            )

            ordem_atribuicao = np.argsort(
                -np.min(distancias_centroides, axis=1)
            )

            novos_clusters_servicos = [[] for _ in range(len(centroides))]
            novos_clusters_indices = [[] for _ in range(len(centroides))]

            for indice_ponto in ordem_atribuicao:

                ordem_centroides = np.argsort(
                    distancias_centroides[indice_ponto]
                )

                servico = servicos[indice_ponto]

                alocado = False

                for indice_centroide in ordem_centroides:

                    servicos_teste = (
                        novos_clusters_servicos[indice_centroide]
                        +
                        [servico]
                    )

                    if estimar_tempo_rota_balanceamento(servicos_teste) <= JORNADA_MAXIMA_SEGUNDOS:

                        novos_clusters_servicos[indice_centroide].append(servico)
                        novos_clusters_indices[indice_centroide].append(indice_ponto)

                        alocado = True
                        break

                if not alocado:

                    # Nenhum cluster existente comporta esta ocorrencia:
                    # cria-se um cluster novo (com um novo centroide, na
                    # propria ocorrencia) em vez de forcar uma alocacao
                    # inviavel. Isso e sinal de que o `k` testado esta mal
                    # dimensionado para este dia.
                    houve_cluster_extra = True

                    log_etapa(
                        f"Fase A (k={k}): nenhum cluster comportou uma "
                        f"ocorrencia - criando cluster extra (k mal "
                        f"dimensionado para este dia)"
                    )

                    novos_clusters_servicos.append([servico])
                    novos_clusters_indices.append([indice_ponto])

                    centroides = np.vstack([
                        centroides,
                        pontos[indice_ponto]
                    ])

                    distancias_centroides = np.sqrt(
                        np.sum(
                            (pontos[:, None, :] - centroides[None, :, :]) ** 2,
                            axis=2
                        )
                    )

            # Descarta clusters vazios e recalcula centroides como a media
            # das coordenadas dos pontos de cada cluster.
            novos_centroides = []
            clusters_servicos_finais = []
            clusters_indices_finais = []

            for indices_cluster, servicos_cluster in zip(
                novos_clusters_indices, novos_clusters_servicos
            ):

                if len(indices_cluster) == 0:
                    continue

                novos_centroides.append(
                    pontos[indices_cluster].mean(axis=0)
                )

                clusters_servicos_finais.append(servicos_cluster)
                clusters_indices_finais.append(indices_cluster)

            centroides_anteriores = centroides
            centroides = np.array(novos_centroides)

            clusters_servicos = clusters_servicos_finais
            clusters_indices = clusters_indices_finais

            # Criterio de estabilizacao: se os centroides praticamente nao
            # se moveram (e o numero de clusters nao mudou), encerra antes
            # do teto de 30 iteracoes.
            if (
                centroides.shape == centroides_anteriores.shape
                and
                np.allclose(centroides, centroides_anteriores, atol=1e-6)
            ):
                break

        equipes_k = []

        for servicos_cluster in clusters_servicos:

            if len(servicos_cluster) == 0:
                continue

            servicos_cluster_ordenados = sorted(
                servicos_cluster,
                key=lambda servico: servico["INCIDENT_DATETIME"]
            )

            equipe = {
                "servicos": servicos_cluster_ordenados
            }

            atualizar_estado_equipe(equipe)

            equipes_k.append(equipe)

        log_etapa(
            f"Particao espacial (fase A, k={k}): {len(equipes_k)} clusters "
            f"formados"
        )

        return equipes_k, houve_cluster_extra

    def _executar_fase_b(equipes):
        """FASE B (refinamento de carga respeitando a geografia): para
        equipes ociosas (abaixo da jornada minima desejada), tenta mover
        ocorrencias apenas para equipes VIZINHAS geograficamente, e so
        aceita o movimento se ele nao aumentar a distancia total estimada
        das duas rotas envolvidas."""

        houve_movimento = True
        iteracoes_refinamento = 0
        movimentos_fase_b = 0

        while houve_movimento and iteracoes_refinamento < 50:

            iteracoes_refinamento += 1
            houve_movimento = False

            centroides_equipes = [
                _centroide_equipe(equipe) for equipe in equipes
            ]

            for indice_origem, equipe_origem in enumerate(equipes):

                if equipe_origem["tempo_total"] >= JORNADA_MINIMA_DESEJADA_SEGUNDOS:
                    continue

                if len(equipes) <= 1:
                    break

                # Vizinhanca geografica: as ate 3 equipes cujo centroide
                # esta mais proximo do centroide da equipe de origem.
                outras = [
                    (indice, math.dist(
                        centroides_equipes[indice_origem],
                        centroides_equipes[indice]
                    ))
                    for indice in range(len(equipes))
                    if indice != indice_origem
                ]

                vizinhas = sorted(
                    outras, key=lambda item: item[1]
                )[:3]

                indices_vizinhos = [indice for indice, _ in vizinhas]

                movimento_aceito = False

                for servico in list(equipe_origem["servicos"]):

                    servicos_origem_restantes = [
                        outro for outro in equipe_origem["servicos"]
                        if outro is not servico
                    ]

                    distancia_origem_antes = estimar_distancia_rota_balanceamento(
                        equipe_origem["servicos"]
                    )

                    distancia_origem_depois = estimar_distancia_rota_balanceamento(
                        servicos_origem_restantes
                    )

                    for indice_destino in indices_vizinhos:

                        equipe_destino = equipes[indice_destino]

                        servicos_teste = (
                            equipe_destino["servicos"]
                            +
                            [servico]
                        )

                        tempo_total_teste = estimar_tempo_rota_balanceamento(
                            servicos_teste
                        )

                        if tempo_total_teste > JORNADA_MAXIMA_SEGUNDOS:
                            continue

                        distancia_destino_antes = estimar_distancia_rota_balanceamento(
                            equipe_destino["servicos"]
                        )

                        distancia_destino_depois = estimar_distancia_rota_balanceamento(
                            servicos_teste
                        )

                        if (
                            distancia_origem_depois + distancia_destino_depois
                            >
                            distancia_origem_antes + distancia_destino_antes
                        ):
                            continue

                        equipe_destino["servicos"].append(servico)
                        atualizar_estado_equipe(equipe_destino)

                        equipe_origem["servicos"] = servicos_origem_restantes

                        if len(equipe_origem["servicos"]) > 0:
                            atualizar_estado_equipe(equipe_origem)
                        else:
                            equipe_origem["tempo_total"] = 0

                        movimentos_fase_b += 1
                        movimento_aceito = True
                        houve_movimento = True

                        break

                    if movimento_aceito:
                        break

                if movimento_aceito:
                    break

            equipes = [
                equipe for equipe in equipes
                if len(equipe["servicos"]) > 0
            ]

        log_etapa(
            f"Refinamento de carga (fase B): {movimentos_fase_b} "
            f"ocorrencias realocadas entre equipes vizinhas em "
            f"{iteracoes_refinamento} iteracoes"
        )

        return equipes

    # ------------------------------------------------------------
    # BUSCA DO MENOR k VIAVEL: comeca no k estimado pela jornada-alvo e
    # tenta reduzir k (menos equipes, cada uma mais cheia) enquanto o
    # resultado ainda tiver equipes ociosas (abaixo da jornada minima
    # desejada) e a alocacao continuar viavel (todas as ocorrencias
    # alocadas, nenhuma jornada acima de 8h). Sem essa busca, o k inicial
    # (dimensionado so pela soma de tempo de servico, sem levar em conta
    # a geografia) tende a superestimar o numero de equipes necessarias,
    # e a absorcao/consolidacao (que agora tem uma tolerancia geografica
    # limitada - ver TOLERANCIA_KM_POR_HORA_OCIOSA) pode nao conseguir
    # compactar tudo sozinha.
    soma_tempo_servico = sum(
        servico["tempo_servico"] for servico in servicos
    )

    k_inicial = max(
        1,
        math.ceil(soma_tempo_servico / JORNADA_ALVO_SEGUNDOS)
    )

    k = k_inicial
    tentativas = 0
    melhor_k = None
    melhor_equipes = None
    passo_busca_ascendente = 1

    while True:

        tentativas += 1

        equipes_tentativa, houve_cluster_extra = _executar_fase_a(k)

        equipes_tentativa = _executar_fase_b(equipes_tentativa)

        equipes_tentativa, equipes_absorvidas_1 = absorver_equipes_subutilizadas(
            equipes_tentativa
        )

        log_etapa(
            f"Compactacao inicial (espacial, k={k}): "
            f"{equipes_absorvidas_1} equipes absorvidas"
        )

        equipes_tentativa = consolidar_equipes(equipes_tentativa)

        equipes_tentativa, equipes_absorvidas_2 = absorver_equipes_subutilizadas(
            equipes_tentativa
        )

        log_etapa(
            f"Compactacao final (espacial, k={k}): "
            f"{equipes_absorvidas_2} equipes absorvidas"
        )

        try:
            _validar_balanceamento(
                df,
                equipes_tentativa,
                rotulo=f"espacial (tentativa k={k})"
            )
            viavel = True
        except RuntimeError as erro_validacao:
            viavel = False
            log_etapa(
                f"Tentativa k={k} inviavel ({erro_validacao}) - "
                f"revertendo para o ultimo k viavel"
            )

        if houve_cluster_extra and melhor_k is not None:
            # Criar cluster extra e sinal de que este k ja esta pequeno
            # demais para a geografia do dia - nao vale a pena continuar
            # reduzindo k a partir daqui.
            viavel = False

        if viavel:

            melhor_k = k
            melhor_equipes = equipes_tentativa
            passo_busca_ascendente = 1

            tem_equipe_ociosa = any(
                equipe["tempo_total"] < JORNADA_MINIMA_DESEJADA_SEGUNDOS
                for equipe in equipes_tentativa
            )

            if not tem_equipe_ociosa or k <= 1:
                break

            k -= 1

        else:
            if melhor_equipes is None:
                # Nenhum k ate agora (nem o k_inicial) produziu uma
                # alocacao viavel - aumenta k (mais equipes, cada uma com
                # menos carga) e tenta de novo, em vez de desistir e deixar
                # a validacao final denunciar uma alocacao que sabemos, ja
                # aqui, que estourou a jornada maxima. Sem isso, um k_inicial
                # mal dimensionado (a estimativa inicial e so pela soma do
                # tempo de servico, sem geografia - ver comentario acima)
                # podia derrubar o pipeline inteiro com um RuntimeError.
                #
                # O passo cresce exponencialmente (1, 2, 4, 8, ...) a cada
                # tentativa ascendente consecutiva: cada tentativa de k já
                # é cara (fase A/B + absorcao + consolidacao sobre dezenas
                # ou centenas de equipes), entao subir k de 1 em 1 quando o
                # k_inicial esta muito subdimensionado (dia com geografia
                # dificil) pode levar dezenas de tentativas lentas ate
                # encontrar a regiao viavel. Assim que uma tentativa e
                # viavel, o passo e resetado (linha abaixo, no ramo `if
                # viavel`) para nao pular k's melhores durante a busca
                # descendente normal.
                if k >= total_servicos:
                    # Ja tentamos k = total_servicos (uma equipe por
                    # ocorrencia, o maximo possivel) e continua inviavel -
                    # so acontece se uma ocorrencia isolada ja excede a
                    # jornada maxima sozinha. Nao ha mais k maior para
                    # tentar.
                    break

                k = min(
                    k + passo_busca_ascendente,
                    total_servicos
                )
                passo_busca_ascendente *= 2

                continue

            break

    if melhor_equipes is None:
        # Nenhum k, nem mesmo k = total_servicos (uma equipe por
        # ocorrencia), produziu uma alocacao viavel - so acontece se uma
        # ocorrencia isolada ja excede a jornada maxima sozinha. Usa o
        # resultado do k inicial mesmo assim, e deixa a validacao final
        # abaixo denunciar o problema com uma mensagem clara, em vez de
        # silenciosamente devolver uma lista vazia.
        melhor_k = k_inicial
        melhor_equipes, _ = _executar_fase_a(k_inicial)
        melhor_equipes = _executar_fase_b(melhor_equipes)
        melhor_equipes, _ = absorver_equipes_subutilizadas(melhor_equipes)
        melhor_equipes = consolidar_equipes(melhor_equipes)
        melhor_equipes, _ = absorver_equipes_subutilizadas(melhor_equipes)

    equipes = melhor_equipes

    equipes = _reparar_equipes_sobrecarregadas(equipes)

    log_etapa(
        f"Busca do menor k viavel: k inicial = {k_inicial}, "
        f"k final = {melhor_k}, {tentativas} tentativa(s)"
    )

    log_etapa(
        f"Balanceamento espacial concluido: {len(equipes)} equipes em "
        f"{time.perf_counter() - inicio_balanceamento:.1f}s"
    )

    _validar_balanceamento(df, equipes, rotulo="espacial")

    return equipes


def alocar_equipes_fifo(df):
    """Aloca as ocorrências às equipes na ordem cronológica de
    abertura, sem qualquer critério de balanceamento (a primeira equipe
    que comportar a ocorrência a recebe). Usada para gerar o cenário
    "Sem balanceamento + FIFO" - a referência de comparação de todo o
    estudo de ablação."""

    equipes = []

    servicos = [
        linha for _, linha in df.sort_values(
            "INCIDENT_DATETIME"
        ).iterrows()
    ]

    for servico in servicos:

        alocado = False

        for equipe in equipes:

            servicos_teste = (
                equipe["servicos"]
                +
                [servico]
            )

            if equipe_consegue_atender(servicos_teste):

                equipe["servicos"].append(servico)

                atualizar_estado_equipe(equipe)

                alocado = True

                break

        if not alocado:

            equipes.append(
                criar_equipe(servico)
            )

    return equipes


def calcular_metricas_balanceamento(equipes):
    """A partir de uma alocação de equipes já definida, calcula os
    indicadores de equilíbrio de carga reportados no TCC (Tabela 5):
    carga média, desvio padrão e coeficiente de variação da carga
    (desvio padrão / média - permite comparar dispersão entre cenários
    com médias diferentes), tempo ocioso total (soma de quanto cada
    equipe ficou abaixo da jornada máxima), diferença entre a equipe
    mais e a menos carregada, jornada excedida (horas acima do limite
    de 8h) e percentual de equipes com baixa alocação (< 7,75h)."""

    cargas = np.array([
        estimar_tempo_rota_balanceamento(equipe["servicos"]) / 3600
        for equipe in equipes
        if len(equipe["servicos"]) > 0
    ])

    if len(cargas) == 0:
        cargas = np.array([0])

    media_carga = cargas.mean()

    desvio_carga = cargas.std(ddof=0)

    coeficiente_variacao = (
        desvio_carga / media_carga
        if media_carga > 0
        else 0
    )

    tempo_ocioso = np.maximum(
        0,
        JORNADA_MAXIMA_HORAS - cargas
    )

    diferenca_carga = cargas.max() - cargas.min()

    jornada_excedida = np.maximum(
        0,
        cargas - JORNADA_MAXIMA_HORAS
    )

    equipes_baixa_alocacao = (
        cargas < JORNADA_MINIMA_DESEJADA_HORAS
    ).sum()

    return {
        "equipes_usadas": len(cargas),
        "carga_media_horas": media_carga,
        "carga_desvio_padrao_horas": desvio_carga,
        "coeficiente_variacao_carga": coeficiente_variacao,
        "tempo_ocioso_total_horas": tempo_ocioso.sum(),
        "tempo_ocioso_medio_horas": tempo_ocioso.mean(),
        "diferenca_mais_menos_carregada_horas": diferenca_carga,
        "jornada_excedida_total_horas": jornada_excedida.sum(),
        "equipes_baixa_alocacao": equipes_baixa_alocacao,
        "percentual_equipes_baixa_alocacao": (
            equipes_baixa_alocacao / len(cargas) * 100
            if len(cargas) > 0
            else 0
        ),
        "cargas_horas": cargas.tolist()
    }


def intervalo_confianca_95(valores):
    """Calcula a margem de erro do intervalo de confiança de 95% (IC95)
    para a média de uma amostra: z * desvio_padrão / sqrt(n), com
    z = 1,96 (aproximação normal, válida para n razoavelmente grande). O
    IC95 indica a faixa em que se espera que a média populacional real
    esteja contida em 95% das repetições do experimento - quanto menor a
    margem, mais precisa a estimativa da média a partir dos 364 dias
    simulados."""

    valores = pd.Series(valores).dropna()

    if len(valores) == 0:
        return 0

    if len(valores) == 1:
        return 0

    return (
        NIVEL_CONFIANCA_Z
        *
        valores.std(ddof=1)
        /
        math.sqrt(len(valores))
    )


def resumir_por_metodo(df, coluna_metodo):
    """Agrega os registros diários (um por dia/cenário) em um resumo
    estatístico por método/cenário: para cada métrica de interesse,
    calcula a média, o desvio padrão e a margem de erro do IC95% ao
    longo dos dias simulados. É a função usada para gerar todas as
    tabelas de "média ± IC95" apresentadas no TCC."""

    registros = []

    if len(df) == 0:
        return pd.DataFrame()

    metricas_resumo = [
        "distancia_total_km",
        "tempo_deslocamento_horas",
        "carga_media_horas",
        "equipes_usadas",
        "carga_desvio_padrao_horas",
        "coeficiente_variacao_carga",
        "tempo_ocioso_total_horas",
        "diferenca_mais_menos_carregada_horas",
        "equipes_baixa_alocacao",
        "percentual_equipes_baixa_alocacao",
        "jornada_excedida_total_horas",
        "rotas_inviaveis",
        "percentual_rotas_inviaveis",
        "funcao_objetivo"
    ]

    for metodo, grupo in df.groupby(coluna_metodo):

        registro = {
            coluna_metodo: metodo,
            "dias": grupo["dia"].nunique()
        }

        for metrica in metricas_resumo:

            if metrica not in grupo.columns:
                continue

            registro[f"{metrica}_media"] = grupo[metrica].mean()
            registro[f"{metrica}_desvio_padrao"] = grupo[metrica].std(ddof=1)
            registro[f"{metrica}_ic95"] = intervalo_confianca_95(
                grupo[metrica]
            )

        registros.append(registro)

    return pd.DataFrame(registros)

# ==========================================================
# ANALISE DE SENSIBILIDADE DOS HIPERPARAMETROS DO ACO
# ==========================================================
# Os cinco hiperparametros do ACO (ACO_FORMIGAS, ACO_ITERACOES, ACO_ALPHA,
# ACO_BETA, ACO_EVAPORACAO) foram fixados com base na literatura, sem
# justificativa experimental própria. Esta rotina NÃO busca a melhor
# configuração (isso seria overfitting na própria instância) - o objetivo
# é demonstrar ROBUSTEZ: mostrar que as conclusões do TCC não dependem
# dessa escolha, variando cada hiperparâmetro individualmente
# (uma-de-cada-vez, não fatorial completo) a partir do padrão e medindo o
# quanto a distância final (ACO + 2OPT) se desvia.
#
# Roda inteiramente à parte do pipeline principal, sobre uma subamostra
# estratificada de dias (para manter o custo computacional controlado) e
# com replicações por semente determinística (para separar o efeito da
# configuração do ruído intrínseco do algoritmo).

EXECUTAR_SENSIBILIDADE_ACO = True

# Numero de dias (dentre os processados) usados na analise de
# sensibilidade. Reduzir para acelerar uma execucao de teste.
N_DIAS_SENSIBILIDADE = 24

# Replicacoes por (dia, equipe, configuracao), com sementes distintas e
# deterministicas - permite estimar o IC95% do desvio de cada configuracao.
N_REPLICACOES = 3

# Configuracoes avaliadas: variacao uma-de-cada-vez a partir do padrao
# (None = usa a constante global ACO_* correspondente).
CONFIGURACOES_SENSIBILIDADE_ACO = [
    {"rotulo": "padrao"},
    {"rotulo": "alpha=0.5", "alpha": 0.5},
    {"rotulo": "alpha=2", "alpha": 2},
    {"rotulo": "beta=1", "beta": 1},
    {"rotulo": "beta=5", "beta": 5},
    {"rotulo": "evaporacao=0.10", "evaporacao": 0.10},
    {"rotulo": "evaporacao=0.60", "evaporacao": 0.60},
    {"rotulo": "orcamento reduzido (10f/20i)", "formigas": 10, "iteracoes": 20},
    {"rotulo": "orcamento ampliado (50f/100i)", "formigas": 50, "iteracoes": 100},
    {"rotulo": "Q=10", "q": 10},
    {"rotulo": "Q=1000", "q": 1000},
]


def semente_sensibilidade(dia, equipe, rotulo, replicacao):
    """Gera uma semente determinística e estável entre execuções do
    script (hash SHA-256 do texto, diferente de hash() nativo do Python,
    que varia entre processos) a partir de (dia, equipe, configuração,
    replicação), para que cada combinação sempre sorteie o mesmo
    resultado do ACO."""

    texto = f"{dia}|{equipe}|{rotulo}|{replicacao}"

    digest = hashlib.sha256(texto.encode("utf-8")).hexdigest()

    return int(digest[:8], 16)


def selecionar_dias_sensibilidade(base, dias, n_dias):
    """Seleciona uma subamostra de `n_dias` dias estratificada pelo
    volume diário de ocorrências: ordena todos os dias pelo volume e
    escolhe índices igualmente espaçados (via np.linspace) nessa lista
    ordenada, garantindo cobertura de toda a faixa observada (não apenas
    a média)."""

    contagem = base.groupby("dia").size().reindex(dias)

    dias_ordenados = contagem.sort_values(kind="stable").index.tolist()

    total = len(dias_ordenados)

    if n_dias >= total:
        return dias_ordenados

    indices = sorted(set(
        int(round(indice))
        for indice in np.linspace(0, total - 1, n_dias)
    ))

    candidatos_restantes = [
        indice for indice in range(total)
        if indice not in indices
    ]

    while len(indices) < n_dias and candidatos_restantes:

        indices.append(candidatos_restantes.pop(0))

        indices = sorted(set(indices))

    indices = sorted(indices)[:n_dias]

    return [dias_ordenados[indice] for indice in indices]


def analisar_sensibilidade_aco():
    """Executa a analise de sensibilidade dos hiperparametros do ACO
    (ver cabecalho da secao acima) e exporta os resultados brutos, o
    resumo estatistico por configuracao e o grafico de desvios. É aditiva
    e roda inteiramente após o pipeline principal: recalcula o
    balanceamento (determinístico, sem aleatoriedade) para os dias
    selecionados apenas para obter os grupos de ocorrências por equipe,
    sem alterar nenhum resultado já exportado."""

    log_etapa("Iniciando analise de sensibilidade dos hiperparametros do ACO")

    inicio_sensibilidade = time.perf_counter()

    dias_selecionados = selecionar_dias_sensibilidade(
        base,
        dias,
        N_DIAS_SENSIBILIDADE
    )

    log_etapa(
        f"Sensibilidade ACO: {len(dias_selecionados)} dias selecionados "
        f"(estratificados por volume diario de ocorrencias)"
    )

    caminho_checkpoint = caminho_sensibilidade(
        "sensibilidade_aco_checkpoint.csv"
    )

    registros_brutos = []

    dias_ja_processados = set()

    if os.path.exists(caminho_checkpoint):

        df_checkpoint = pd.read_csv(caminho_checkpoint)

        registros_brutos = df_checkpoint.to_dict("records")

        dias_ja_processados = set(
            df_checkpoint["dia"].astype(str)
        )

        log_etapa(
            f"Sensibilidade ACO: checkpoint encontrado com "
            f"{len(registros_brutos)} execucao(oes) de "
            f"{len(dias_ja_processados)} dia(s) - retomando de onde parou"
        )

    total_execucoes = len(registros_brutos)

    for dia in dias_selecionados:

        if str(dia) in dias_ja_processados:

            log_etapa(
                f"Sensibilidade ACO: dia {dia} ja esta no checkpoint - "
                f"pulando"
            )

            continue

        df_dia = base[base["dia"] == dia].copy()

        if len(df_dia) < 2:
            continue

        equipes = alocar_equipes_balanceado(df_dia)

        registros_dia = []

        for equipe_id, equipe in enumerate(equipes):

            df_eq = pd.DataFrame(
                equipe["servicos"]
            ).reset_index(drop=True)

            if len(df_eq) < 3:
                # Rota com <= 2 pontos e trivial - nao ha ordem a
                # otimizar, logo nao e sensivel aos hiperparametros do ACO.
                continue

            coords = list(
                zip(df_eq["lat"], df_eq["lon"])
            )

            matriz = criar_matriz(coords)

            for config in CONFIGURACOES_SENSIBILIDADE_ACO:

                for replicacao in range(N_REPLICACOES):

                    semente = semente_sensibilidade(
                        dia,
                        equipe_id,
                        config["rotulo"],
                        replicacao
                    )

                    inicio_execucao = time.perf_counter()

                    path_aco = otimizar_rota_aco(
                        matriz,
                        formigas=config.get("formigas"),
                        iteracoes=config.get("iteracoes"),
                        alpha=config.get("alpha"),
                        beta=config.get("beta"),
                        evaporacao=config.get("evaporacao"),
                        q=config.get("q"),
                        semente=semente
                    )

                    tempo_execucao_aco = (
                        time.perf_counter() - inicio_execucao
                    )

                    distancia_aco = distancia_rota(path_aco, matriz)

                    path_2opt = melhorar_rota_2opt(path_aco, matriz)

                    distancia_2opt = distancia_rota(path_2opt, matriz)

                    registro = {
                        "dia": dia,
                        "equipe": equipe_id,
                        "configuracao": config["rotulo"],
                        "replicacao": replicacao,
                        "quantidade_pontos": len(df_eq),
                        "distancia_aco_km": distancia_aco,
                        "distancia_aco_2opt_km": distancia_2opt,
                        "tempo_execucao_aco_segundos": tempo_execucao_aco
                    }

                    registros_brutos.append(registro)
                    registros_dia.append(registro)

                    total_execucoes += 1

        # Checkpoint incremental: grava as execucoes deste dia no disco
        # assim que ele termina, para que uma interrupcao (erro, queda de
        # energia, fechamento do terminal) nao obrigue a reprocessar dias
        # ja concluidos - na proxima execucao, `dias_ja_processados` acima
        # pula direto para o primeiro dia pendente.
        if len(registros_dia) > 0:

            df_checkpoint_dia = pd.DataFrame(registros_dia)

            os.makedirs(
                os.path.dirname(caminho_checkpoint),
                exist_ok=True
            )

            df_checkpoint_dia.to_csv(
                caminho_checkpoint,
                mode="a",
                header=not os.path.exists(caminho_checkpoint),
                index=False
            )

        log_etapa(
            f"Sensibilidade ACO: dia {dia} concluido "
            f"({total_execucoes} execucoes acumuladas) - checkpoint salvo"
        )

    df_sensibilidade = pd.DataFrame(registros_brutos)

    df_sensibilidade.to_excel(
        caminho_sensibilidade("sensibilidade_aco_bruto.xlsx"),
        index=False
    )

    # Todos os dias selecionados foram processados e o resultado final ja
    # foi exportado - o checkpoint deixa de ser necessario. Remove-lo evita
    # que uma proxima execucao (com `dias_selecionados` diferente, por
    # exemplo apos mudar N_DIAS_SENSIBILIDADE ou a base de dados) pule dias
    # por engano usando um checkpoint de uma analise anterior ja concluida.
    if os.path.exists(caminho_checkpoint):
        os.remove(caminho_checkpoint)

    if len(df_sensibilidade) == 0:

        log_etapa(
            "Sensibilidade ACO: nenhum registro gerado, resumo e grafico "
            "nao foram criados"
        )

        return

    # Desvio percentual de cada execucao em relacao ao padrao NA MESMA
    # replicacao (mesma semente-base de dia/equipe/replicacao), isolando o
    # efeito da configuracao do ruido intrinseco do algoritmo.
    # v11: o desvio e calculado para as DUAS etapas - ACO isolado
    # (distancia_aco_km) e ACO + 2-opt (distancia_aco_2opt_km) -, tanto
    # como media dos desvios por rota quanto sobre a distancia total
    # agregada. As colunas sem sufixo (desvio_percentual_*) mantem o
    # significado da v10: ACO + 2-opt, media por rota.
    chaves = ["dia", "equipe", "replicacao"]

    padrao = df_sensibilidade[
        df_sensibilidade["configuracao"] == "padrao"
    ][
        chaves + ["distancia_aco_km", "distancia_aco_2opt_km"]
    ].rename(
        columns={
            "distancia_aco_km": "distancia_padrao_aco_km",
            "distancia_aco_2opt_km": "distancia_padrao_km"
        }
    )

    df_comparado = df_sensibilidade.merge(
        padrao,
        on=chaves,
        how="left"
    )

    df_comparado["desvio_percentual"] = (
        (df_comparado["distancia_aco_2opt_km"] - df_comparado["distancia_padrao_km"])
        / df_comparado["distancia_padrao_km"]
        * 100
    )

    df_comparado["desvio_percentual_aco"] = (
        (df_comparado["distancia_aco_km"] - df_comparado["distancia_padrao_aco_km"])
        / df_comparado["distancia_padrao_aco_km"]
        * 100
    )

    total_padrao_2opt = df_comparado["distancia_padrao_km"].sum()
    total_padrao_aco = df_comparado["distancia_padrao_aco_km"].sum()

    linhas_resumo = []

    for configuracao, grupo in df_comparado.groupby("configuracao"):

        desvios = grupo["desvio_percentual"].dropna()
        desvios_aco = grupo["desvio_percentual_aco"].dropna()

        linhas_resumo.append({
            "configuracao": configuracao,
            "execucoes": len(grupo),
            # ACO + 2-opt, media dos desvios por rota (mesmo significado da v10)
            "desvio_percentual_medio": desvios.mean(),
            "desvio_percentual_ic95": intervalo_confianca_95(desvios),
            "desvio_percentual_maximo_absoluto": desvios.abs().max(),
            # ACO + 2-opt, sobre a distancia total agregada
            "desvio_percentual_agregado_aco_2opt": (
                (grupo["distancia_aco_2opt_km"].sum() / grupo["distancia_padrao_km"].sum() - 1)
                * 100
            ),
            # ACO isolado (antes do 2-opt)
            "desvio_percentual_medio_aco": desvios_aco.mean(),
            "desvio_percentual_ic95_aco": intervalo_confianca_95(desvios_aco),
            "desvio_percentual_agregado_aco": (
                (grupo["distancia_aco_km"].sum() / grupo["distancia_padrao_aco_km"].sum() - 1)
                * 100
            ),
            "tempo_medio_execucao_segundos": grupo[
                "tempo_execucao_aco_segundos"
            ].mean(),
            "tempo_total_execucao_segundos": grupo[
                "tempo_execucao_aco_segundos"
            ].sum()
        })

    df_resumo_sensibilidade = pd.DataFrame(linhas_resumo).sort_values(
        "desvio_percentual_medio",
        key=lambda serie: serie.abs()
    )

    # Tempo de execucao do ACO (configuracao padrao) por faixa de tamanho
    # da rota - insumo para a discussao de escalabilidade.
    df_padrao_tempo = df_sensibilidade[
        df_sensibilidade["configuracao"] == "padrao"
    ].copy()

    if len(df_padrao_tempo) > 0:

        df_padrao_tempo["faixa_pontos"] = pd.cut(
            df_padrao_tempo["quantidade_pontos"],
            bins=[0, 10, 20, 30, 40, 1000],
            labels=["ate 10", "11 a 20", "21 a 30", "31 a 40", "acima de 40"]
        )

        df_padrao_tempo.groupby("faixa_pontos", observed=True).agg(
            rotas=("tempo_execucao_aco_segundos", "size"),
            pontos_medio=("quantidade_pontos", "mean"),
            tempo_medio_aco_segundos=("tempo_execucao_aco_segundos", "mean"),
            tempo_p90_aco_segundos=(
                "tempo_execucao_aco_segundos",
                lambda serie: serie.quantile(0.9)
            )
        ).reset_index().to_excel(
            caminho_sensibilidade("sensibilidade_aco_tempo_por_tamanho.xlsx"),
            index=False
        )

    df_resumo_sensibilidade.to_excel(
        caminho_sensibilidade("sensibilidade_aco_resumo.xlsx"),
        index=False
    )

    if plt is not None:

        df_plot = df_resumo_sensibilidade.sort_values(
            "desvio_percentual_medio"
        )

        cores = [
            "#1B7F79" if configuracao == "padrao" else "#9AA3AB"
            for configuracao in df_plot["configuracao"]
        ]

        plt.figure(figsize=(9, 5))

        eixo = plt.gca()

        eixo.barh(
            df_plot["configuracao"],
            df_plot["desvio_percentual_medio"],
            xerr=df_plot["desvio_percentual_ic95"],
            color=cores,
            capsize=4
        )

        eixo.axvline(
            0,
            color="#1B7F79",
            linewidth=1.5
        )

        eixo.spines["top"].set_visible(False)
        eixo.spines["right"].set_visible(False)

        eixo.set_xlabel(
            "Desvio percentual medio da distancia vs. configuracao "
            "padrao (%)"
        )

        eixo.set_title("Sensibilidade dos hiperparametros do ACO")

        plt.tight_layout()

        plt.savefig(
            caminho_sensibilidade("grafico_sensibilidade_aco.png"),
            dpi=150
        )

        plt.close()

    else:

        log_etapa(
            "matplotlib nao encontrado; grafico de sensibilidade nao "
            "foi gerado"
        )

    desvio_maximo_geral = df_resumo_sensibilidade[
        df_resumo_sensibilidade["configuracao"] != "padrao"
    ]["desvio_percentual_maximo_absoluto"].max()

    log_etapa(
        f"Sensibilidade ACO concluida em "
        f"{time.perf_counter() - inicio_sensibilidade:.1f}s - "
        f"{total_execucoes} execucoes do ACO - "
        f"desvio maximo entre configuracoes: {desvio_maximo_geral:.2f}%"
    )


# ==========================================================
# ANALISE DE SENSIBILIDADE DA TOLERANCIA DISTANCIA/OCIOSIDADE (v11)
# ==========================================================
# TOLERANCIA_KM_POR_HORA_OCIOSA controla quantos km adicionais de
# deslocamento o balanceamento aceita para eliminar uma hora de
# ociosidade. Esta rotina reexecuta o balanceamento espacial para uma
# grade de tolerancias, sobre a MESMA subamostra estratificada de dias da
# sensibilidade do ACO, e mede: equipes utilizadas, indicadores de carga
# (planejada) e distancia total com roteamento ACO + 2-opt (semente
# deterministica). Substitui a planilha avulsa da v10, que nao era gerada
# pelo script. A grade vai alem de 60 para mostrar se o valor adotado
# esta ou nao na borda do intervalo testado.
EXECUTAR_SENSIBILIDADE_TOLERANCIA = True

GRADE_TOLERANCIA_KM_POR_HORA_OCIOSA = [0, 10, 20, 40, 60, 80, 120, 200]


def analisar_sensibilidade_tolerancia():

    global TOLERANCIA_KM_POR_HORA_OCIOSA

    log_etapa("Iniciando analise de sensibilidade da tolerancia distancia/ociosidade")

    inicio_analise = time.perf_counter()

    tolerancia_original = TOLERANCIA_KM_POR_HORA_OCIOSA

    dias_selecionados = selecionar_dias_sensibilidade(
        base,
        dias,
        N_DIAS_SENSIBILIDADE
    )

    registros = []

    try:

        for tolerancia in GRADE_TOLERANCIA_KM_POR_HORA_OCIOSA:

            TOLERANCIA_KM_POR_HORA_OCIOSA = float(tolerancia)

            for dia in dias_selecionados:

                df_dia = base[base["dia"] == dia].copy()

                if len(df_dia) < 2:
                    continue

                try:
                    equipes_tol = alocar_equipes_balanceado(df_dia)
                except Exception as erro_tol:
                    log_etapa(
                        f"Sensibilidade tolerancia {tolerancia}: dia {dia} "
                        f"ignorado ({erro_tol})"
                    )
                    continue

                indicadores = calcular_metricas_balanceamento(equipes_tol)

                distancia_total = 0.0
                jornada_excedida_rotas = 0.0

                for indice_equipe, equipe_tol in enumerate(equipes_tol):

                    df_eq_tol = pd.DataFrame(
                        equipe_tol["servicos"]
                    ).reset_index(drop=True)

                    coords_tol = list(zip(df_eq_tol["lat"], df_eq_tol["lon"]))

                    matriz_tol = criar_matriz(coords_tol)

                    if len(coords_tol) <= 2:
                        path_tol = list(range(len(coords_tol)))
                    else:
                        path_tol = melhorar_rota_2opt(
                            otimizar_rota_aco(
                                matriz_tol,
                                semente=semente_sensibilidade(
                                    dia, indice_equipe, "tolerancia", 0
                                )
                            ),
                            matriz_tol
                        )

                    metrica_tol = avaliar_caminho(
                        path_tol, df_eq_tol, coords_tol, matriz_tol
                    )

                    distancia_total += metrica_tol["distancia_total_km"]
                    jornada_excedida_rotas += metrica_tol["jornada_excedida_horas"]

                registros.append({
                    "tolerancia_km_por_hora": tolerancia,
                    "dia": dia,
                    "equipes_usadas": indicadores["equipes_usadas"],
                    "carga_media_horas": indicadores["carga_media_horas"],
                    "coeficiente_variacao_carga": indicadores["coeficiente_variacao_carga"],
                    "tempo_ocioso_total_horas": indicadores["tempo_ocioso_total_horas"],
                    "diferenca_mais_menos_carregada_horas": indicadores[
                        "diferenca_mais_menos_carregada_horas"
                    ],
                    "percentual_equipes_baixa_alocacao": indicadores[
                        "percentual_equipes_baixa_alocacao"
                    ],
                    "distancia_km": distancia_total,
                    "jornada_excedida_rotas_horas": jornada_excedida_rotas,
                    "funcao_objetivo": calcular_funcao_objetivo(
                        distancia_total,
                        jornada_excedida_rotas,
                        indicadores["tempo_ocioso_total_horas"],
                        indicadores["diferenca_mais_menos_carregada_horas"]
                    )
                })

            log_etapa(
                f"Sensibilidade tolerancia: {tolerancia} km/h ociosa concluida"
            )

    finally:

        TOLERANCIA_KM_POR_HORA_OCIOSA = tolerancia_original

    if len(registros) == 0:
        log_etapa("Sensibilidade tolerancia: nenhum registro gerado")
        return

    df_tol = pd.DataFrame(registros)

    df_tol.to_excel(
        caminho_sensibilidade("sensibilidade_tolerancia_ociosidade_bruto.xlsx"),
        index=False
    )

    colunas_resumo_tol = [
        "equipes_usadas",
        "carga_media_horas",
        "coeficiente_variacao_carga",
        "tempo_ocioso_total_horas",
        "percentual_equipes_baixa_alocacao",
        "distancia_km",
        "funcao_objetivo"
    ]

    resumo_tol = df_tol.groupby("tolerancia_km_por_hora").agg(
        dias=("dia", "nunique"),
        **{f"{coluna}_media": (coluna, "mean") for coluna in colunas_resumo_tol},
        **{
            f"{coluna}_ic95": (coluna, intervalo_confianca_95)
            for coluna in ("distancia_km", "tempo_ocioso_total_horas")
        }
    ).reset_index()

    resumo_tol.to_excel(
        caminho_tabela("sensibilidade_tolerancia_ociosidade.xlsx"),
        index=False
    )

    log_etapa(
        f"Sensibilidade da tolerancia concluida em "
        f"{time.perf_counter() - inicio_analise:.1f}s"
    )

# ==========================================================
# ESTRUTURAS
# ==========================================================

rotas_json = []

metricas = []

sequencias = []

metricas_baselines = []

metricas_balanceamento = []

metricas_ablation = []

# Totais diarios por metodo de roteamento (TODOS os metodos, incluindo
# Simulated Annealing e OR-Tools) - fonte de `resumo_baselines*.xlsx`.
# Diferente de `metricas_ablation`, que fica restrita aos 4 cenarios
# originais do estudo de ablacao.
metricas_baselines_diario = []

cargas_equipes_boxplot = []

comparativo_2opt = []

# Comparacao pareada, dia a dia, entre o balanceamento antigo
# (`alocar_equipes_bin_packing`, roteirizado so com ACO+2opt) e o novo
# balanceamento espacial (`alocar_equipes_balanceado`, cujo resultado
# ACO+2opt ja e calculado no loop principal) - ver secao 2 das instrucoes
# e `saidas/tabelas/comparativo_balanceamento.xlsx`.
comparativo_balanceamento = []

# Registros de dispersao das replicacoes do ACO no pipeline principal
# (ver N_REPLICACOES_PIPELINE e `avaliar_baselines_rotas`) - uma linha
# por (dia, equipe, replicacao).
REGISTROS_REPLICACOES_ACO_PIPELINE = []

# ==========================================================
# PROCESSAMENTO
# ==========================================================
# Loop principal: repete, para cada um dos dias selecionados, todo o
# pipeline de simulação:
#
#   1) aloca as equipes SEM balanceamento (FIFO puro) e COM
#      balanceamento (`alocar_equipes_fifo` e `alocar_equipes_balanceado`);
#   2) para a alocação sem balanceamento, avalia apenas o método FIFO
#      de roteamento (usado como referência "Sem balanceamento + FIFO");
#   3) para cada equipe da alocação COM balanceamento, compara os 6
#      métodos de roteamento (FIFO, Nearest Neighbor, ACO sem 2-opt,
#      ACO + 2-opt, Simulated Annealing e OR-Tools) e usa o resultado do
#      ACO + 2-opt para desenhar a rota no mapa HTML;
#   3b) roda, isoladamente, o comparativo com o balanceamento
#      bin-packing (falhas nele nao descartam o dia - v11);
#   4) acumula as métricas de cada combinação (cenário x dia) nas
#      listas globais definidas na seção "ESTRUTURAS", que serão
#      convertidas em DataFrames e exportadas mais adiante.

for dia in dias:

    _checkpoint_path = os.path.join(PASTA_SAIDA, "checkpoint", f"{dia}.pkl")

    if not REPROCESSAR_TUDO and os.path.exists(_checkpoint_path):

        log_etapa(f"Dia {dia}: checkpoint encontrado, pulando processamento")

        with open(_checkpoint_path, "rb") as _f_checkpoint:
            _dados_checkpoint = pickle.load(_f_checkpoint)

        rotas_json.extend(_dados_checkpoint["rotas_json"])
        metricas.extend(_dados_checkpoint["metricas"])
        sequencias.extend(_dados_checkpoint["sequencias"])
        metricas_baselines.extend(_dados_checkpoint["metricas_baselines"])
        metricas_balanceamento.extend(_dados_checkpoint["metricas_balanceamento"])
        metricas_ablation.extend(_dados_checkpoint["metricas_ablation"])
        metricas_baselines_diario.extend(_dados_checkpoint["metricas_baselines_diario"])
        cargas_equipes_boxplot.extend(_dados_checkpoint["cargas_equipes_boxplot"])
        comparativo_2opt.extend(_dados_checkpoint["comparativo_2opt"])
        comparativo_balanceamento.extend(_dados_checkpoint["comparativo_balanceamento"])
        REGISTROS_REPLICACOES_ACO_PIPELINE.extend(
            _dados_checkpoint["registros_replicacoes_aco_pipeline"]
        )

        continue

    _snap_rotas_json = len(rotas_json)
    _snap_metricas = len(metricas)
    _snap_sequencias = len(sequencias)
    _snap_metricas_baselines = len(metricas_baselines)
    _snap_metricas_balanceamento = len(metricas_balanceamento)
    _snap_metricas_ablation = len(metricas_ablation)
    _snap_metricas_baselines_diario = len(metricas_baselines_diario)
    _snap_cargas_equipes_boxplot = len(cargas_equipes_boxplot)
    _snap_comparativo_2opt = len(comparativo_2opt)
    _snap_comparativo_balanceamento = len(comparativo_balanceamento)
    _snap_replicacoes_aco = len(REGISTROS_REPLICACOES_ACO_PIPELINE)

    try:


        inicio_dia = time.perf_counter()

        print("\n================================")
        print(f"PROCESSANDO DIA {dia}")
        print("================================")

        df_dia = base[
            base["dia"] == dia
        ].copy()

        if len(df_dia) < 2:
            continue

        log_etapa(
            f"Dia {dia}: {len(df_dia)} ocorrencias para processar"
        )

        print("\nASSOCIANDO NODES...")

        inicio = time.perf_counter()

        df_dia["node"] = ox.distance.nearest_nodes(

            G,

            X=df_dia["lon"],

            Y=df_dia["lat"]
        )

        log_etapa(
            f"Dia {dia}: nodes associados em "
            f"{time.perf_counter() - inicio:.1f}s"
        )

        equipes_fifo = alocar_equipes_fifo(df_dia)

        balanceamento_fifo = calcular_metricas_balanceamento(
            equipes_fifo
        )

        equipes = alocar_equipes_balanceado(df_dia)

        balanceamento_final = calcular_metricas_balanceamento(
            equipes
        )

        for nome_cenario, dados_balanceamento in [
            ("Sem balanceamento", balanceamento_fifo),
            ("Com balanceamento", balanceamento_final)
        ]:

            metricas_balanceamento.append({
                "dia": dia,
                "cenario": nome_cenario,
                "equipes_usadas": dados_balanceamento["equipes_usadas"],
                "carga_media_horas": dados_balanceamento["carga_media_horas"],
                "carga_desvio_padrao_horas": dados_balanceamento[
                    "carga_desvio_padrao_horas"
                ],
                "coeficiente_variacao_carga": dados_balanceamento[
                    "coeficiente_variacao_carga"
                ],
                "tempo_ocioso_total_horas": dados_balanceamento[
                    "tempo_ocioso_total_horas"
                ],
                "tempo_ocioso_medio_horas": dados_balanceamento[
                    "tempo_ocioso_medio_horas"
                ],
                "diferenca_mais_menos_carregada_horas": dados_balanceamento[
                    "diferenca_mais_menos_carregada_horas"
                ],
                "jornada_excedida_total_horas": dados_balanceamento[
                    "jornada_excedida_total_horas"
                ],
                "equipes_baixa_alocacao": dados_balanceamento[
                    "equipes_baixa_alocacao"
                ],
                "percentual_equipes_baixa_alocacao": dados_balanceamento[
                    "percentual_equipes_baixa_alocacao"
                ]
            })

            for carga in dados_balanceamento["cargas_horas"]:

                cargas_equipes_boxplot.append({
                    "dia": dia,
                    "cenario": nome_cenario,
                    "carga_horas": carga
                })

        acumulado_baselines_dia = {}

        # Jornada (horas) de cada equipe do balanceamento ESPACIAL, medida com
        # o metodo oficial ACO + 2OPT - usada em `comparativo_balanceamento.xlsx`
        # (jornada_media_h_espacial / jornada_dp_h_espacial).
        jornadas_espacial_aco2opt = []

        sem_balanceamento_fifo_dia = {
            "distancia_total_km": 0,
            "tempo_deslocamento_horas": 0,
            "tempo_total_horas": 0,
            "quantidade_servicos": 0,
            "jornada_excedida_horas": 0,
            "rotas_inviaveis": 0
        }

        for equipe_fifo in equipes_fifo:

            df_eq_fifo = pd.DataFrame(
                equipe_fifo["servicos"]
            ).reset_index(drop=True)

            coords_fifo = list(
                zip(
                    df_eq_fifo["lat"],
                    df_eq_fifo["lon"]
                )
            )

            matriz_fifo = criar_matriz(coords_fifo)

            path_fifo = caminho_fifo(df_eq_fifo)

            metrica_fifo = avaliar_caminho(
                path_fifo,
                df_eq_fifo,
                coords_fifo,
                matriz_fifo
            )

            sem_balanceamento_fifo_dia["distancia_total_km"] += (
                metrica_fifo["distancia_total_km"]
            )

            sem_balanceamento_fifo_dia["tempo_deslocamento_horas"] += (
                metrica_fifo["tempo_deslocamento"] / 3600
            )

            sem_balanceamento_fifo_dia["tempo_total_horas"] += (
                metrica_fifo["tempo_total"] / 3600
            )

            sem_balanceamento_fifo_dia["jornada_excedida_horas"] += (
                metrica_fifo["jornada_excedida_horas"]
            )

            sem_balanceamento_fifo_dia["rotas_inviaveis"] += int(
                not metrica_fifo["rota_viavel"]
            )

            sem_balanceamento_fifo_dia["quantidade_servicos"] += len(df_eq_fifo)

        print(
            f"\nEQUIPES CRIADAS: {len(equipes)}"
        )

        for equipe_id, equipe in enumerate(equipes):

            print("\n----------------------")
            print(f"EQUIPE {equipe_id}")
            print("----------------------")

            df_eq = pd.DataFrame(
                equipe["servicos"]
            )

            if len(df_eq) < 2:

                df_eq = df_eq.reset_index(drop=True)

                coords = list(
                    zip(
                        df_eq["lat"],
                        df_eq["lon"]
                    )
                )

                matriz = criar_matriz(coords)

                resultados_baselines = {
                    metodo: {
                        "path": list(range(len(df_eq))),
                        "metricas": avaliar_caminho(
                            list(range(len(df_eq))),
                            df_eq,
                            coords,
                            matriz
                        )
                    }
                    for metodo in [
                        "FIFO",
                        "Nearest Neighbor",
                        "ACO sem 2OPT",
                        "ACO + 2OPT",
                        "Simulated Annealing",
                        "OR-Tools"
                    ]
                    if metodo != "OR-Tools" or ORTOOLS_DISPONIVEL
                }

                for metodo_baseline, dados_baseline in resultados_baselines.items():

                    metricas_baseline = dados_baseline["metricas"]

                    if metodo_baseline not in acumulado_baselines_dia:

                        acumulado_baselines_dia[metodo_baseline] = {
                            "distancia_total_km": 0,
                            "tempo_deslocamento_horas": 0,
                            "tempo_total_horas": 0,
                            "quantidade_servicos": 0,
                            "jornada_excedida_horas": 0,
                            "rotas_inviaveis": 0
                        }

                    acumulado_baselines_dia[metodo_baseline][
                        "tempo_total_horas"
                    ] += (
                        metricas_baseline["tempo_total"] / 3600
                    )

                    acumulado_baselines_dia[metodo_baseline][
                        "quantidade_servicos"
                    ] += len(df_eq)

                    acumulado_baselines_dia[metodo_baseline][
                        "jornada_excedida_horas"
                    ] += metricas_baseline["jornada_excedida_horas"]

                    acumulado_baselines_dia[metodo_baseline][
                        "rotas_inviaveis"
                    ] += int(not metricas_baseline["rota_viavel"])

                    metricas_baselines.append({
                        "dia": dia,
                        "equipe": equipe_id,
                        "metodo": metodo_baseline,
                        "distancia_total_km": 0,
                        "tempo_deslocamento_horas": 0,
                        "tempo_total_horas": round(
                            metricas_baseline["tempo_total"] / 3600,
                            2
                        ),
                        "jornada_excedida_horas": round(
                            metricas_baseline["jornada_excedida_horas"],
                            4
                        ),
                        "quantidade_servicos": len(df_eq),
                        "rota_viavel": metricas_baseline["rota_viavel"],
                        "tempo_execucao_segundos": 0.0
                    })

                jornadas_espacial_aco2opt.append(
                    resultados_baselines["ACO + 2OPT"]["metricas"]["jornada_horas"]
                )

                log_etapa(
                    f"Dia {dia} equipe {equipe_id}: ignorada "
                    f"por ter apenas {len(df_eq)} ocorrencia"
                )
                continue

            df_eq = df_eq.reset_index(drop=True)

            coords = list(

                zip(
                    df_eq["lat"],
                    df_eq["lon"]
                )
            )

            log_etapa(
                f"Dia {dia} equipe {equipe_id}: "
                f"{len(df_eq)} ocorrencias, "
                f"tempo alocado {equipe['tempo_total'] / 3600:.2f}h"
            )

            print("CALCULANDO BASELINES DE ROTA...")

            inicio = time.perf_counter()

            resultados_baselines, matriz = avaliar_baselines_rotas(
                df_eq,
                coords,
                dia=dia,
                equipe_id=equipe_id
            )

            metodo_otimizacao = "ACO + 2OPT"

            path = resultados_baselines[
                metodo_otimizacao
            ]["path"]

            log_etapa(
                f"Dia {dia} equipe {equipe_id}: baselines finalizados em "
                f"{time.perf_counter() - inicio:.1f}s"
            )

            for metodo_baseline, dados_baseline in resultados_baselines.items():

                metricas_baseline = dados_baseline["metricas"]

                if metodo_baseline not in acumulado_baselines_dia:

                    acumulado_baselines_dia[metodo_baseline] = {
                        "distancia_total_km": 0,
                        "tempo_deslocamento_horas": 0,
                        "tempo_total_horas": 0,
                        "quantidade_servicos": 0,
                        "jornada_excedida_horas": 0,
                        "rotas_inviaveis": 0
                    }

                acumulado_baselines_dia[metodo_baseline]["distancia_total_km"] += (
                    metricas_baseline["distancia_total_km"]
                )

                acumulado_baselines_dia[metodo_baseline][
                    "tempo_deslocamento_horas"
                ] += (
                    metricas_baseline["tempo_deslocamento"] / 3600
                )

                acumulado_baselines_dia[metodo_baseline]["tempo_total_horas"] += (
                    metricas_baseline["tempo_total"] / 3600
                )

                acumulado_baselines_dia[metodo_baseline][
                    "quantidade_servicos"
                ] += len(df_eq)

                acumulado_baselines_dia[metodo_baseline][
                    "jornada_excedida_horas"
                ] += metricas_baseline["jornada_excedida_horas"]

                acumulado_baselines_dia[metodo_baseline][
                    "rotas_inviaveis"
                ] += int(not metricas_baseline["rota_viavel"])

                metricas_baselines.append({
                    "dia": dia,
                    "equipe": equipe_id,
                    "metodo": metodo_baseline,
                    "distancia_total_km": round(
                        metricas_baseline["distancia_total_km"],
                        2
                    ),
                    "tempo_deslocamento_horas": round(
                        metricas_baseline["tempo_deslocamento"] / 3600,
                        2
                    ),
                    "tempo_total_horas": round(
                        metricas_baseline["tempo_total"] / 3600,
                        2
                    ),
                    "jornada_excedida_horas": round(
                        metricas_baseline["jornada_excedida_horas"],
                        4
                    ),
                    "quantidade_servicos": len(df_eq),
                    "rota_viavel": metricas_baseline["rota_viavel"],
                    "tempo_execucao_segundos": metricas_baseline[
                        "tempo_execucao_segundos"
                    ]
                })

            distancia_aco = resultados_baselines[
                "ACO sem 2OPT"
            ]["metricas"]["distancia_total_km"]

            distancia_2opt = resultados_baselines[
                "ACO + 2OPT"
            ]["metricas"]["distancia_total_km"]

            comparativo_2opt.append({
                "dia": dia,
                "equipe": equipe_id,
                "antes_2opt_km": distancia_aco,
                "depois_2opt_km": distancia_2opt,
                "ganho_2opt_percentual": (
                    (distancia_aco - distancia_2opt)
                    / distancia_aco
                    * 100
                    if distancia_aco > 0
                    else 0
                )
            })

            metrica_final = resultados_baselines[
                metodo_otimizacao
            ]["metricas"]

            distancia_total = metrica_final["distancia_total_km"]

            tempo_deslocamento = metrica_final["tempo_deslocamento"]

            tempo_servicos = (
                df_eq["tempo_servico"].sum()
            )

            tempo_buffer_total = (

                len(df_eq)

                *

                TEMPO_BUFFER_MINUTOS

                * 60
            )

            tempo_total = metrica_final["tempo_total"]

            tempo_total_com_buffer = (

                tempo_total

                +

                tempo_buffer_total
            )

            jornada_horas = metrica_final["jornada_horas"]

            jornadas_espacial_aco2opt.append(jornada_horas)

            rota_viavel = metrica_final["rota_viavel"]

            rota_coords = []

            log_etapa(
                f"Dia {dia} equipe {equipe_id}: calculando "
                f"{max(len(path) - 1, 0)} trechos na malha viaria"
            )

            inicio = time.perf_counter()

            for i in range(len(path) - 1):

                origem = df_eq.iloc[
                    path[i]
                ]["node"]

                destino = df_eq.iloc[
                    path[i + 1]
                ]["node"]

                try:

                    route = ox.shortest_path(

                        G,

                        origem,

                        destino,

                        weight="length"
                    )

                    if route is None:
                        continue

                    segmento = []

                    for node in route:

                        segmento.append([

                            G.nodes[node]["y"],
                            G.nodes[node]["x"]

                        ])

                    if len(rota_coords) > 0:
                        segmento = segmento[1:]

                    rota_coords.extend(
                        segmento
                    )

                except Exception:

                    continue

                if (
                    (i + 1) % 10 == 0
                    or
                    i + 1 == len(path) - 1
                ):

                    log_etapa(
                        f"Dia {dia} equipe {equipe_id}: "
                        f"{i + 1}/{len(path) - 1} trechos calculados"
                    )

            log_etapa(
                f"Dia {dia} equipe {equipe_id}: trechos finalizados em "
                f"{time.perf_counter() - inicio:.1f}s"
            )

            if len(rota_coords) == 0:

                rota_coords = [

                    [float(lat), float(lon)]

                    for lat, lon in zip(
                        df_eq["lat"],
                        df_eq["lon"]
                    )
                ]

            pontos = []

            for ordem, idx in enumerate(path):

                linha = df_eq.iloc[idx]

                if ordem == 0:
                    tempo_deslocamento_atendimento = 0
                else:
                    idx_anterior = path[ordem - 1]

                    dist_atendimento = distancia_km(

                        coords[idx_anterior][0],
                        coords[idx_anterior][1],

                        coords[idx][0],
                        coords[idx][1]
                    )

                    tempo_deslocamento_atendimento = (

                        dist_atendimento

                        / VELOCIDADE_MEDIA

                    ) * 3600

                pontos.append({

                    "ordem": ordem + 1,

                    "zip": str(
                        linha["zip"]
                    ),

                    "lat": float(
                        linha["lat"]
                    ),

                    "lon": float(
                        linha["lon"]
                    )
                })

                sequencias.append({

                    "dia": dia,

                    "equipe": equipe_id,

                    "ordem_atendimento": ordem + 1,

                    "metodo_otimizacao": metodo_otimizacao,

                    "id_ocorrencia": linha[
                        COLUNA_ID_OCORRENCIA
                    ],

                    "linha_base_excel": linha[
                        "linha_base_excel"
                    ],

                    "zip": linha["zip"],

                    "lat": linha["lat"],

                    "lon": linha["lon"],

                    "tempo_servico": linha[
                        "tempo_servico"
                    ],

                    "tempo_deslocamento": tempo_deslocamento_atendimento,

                    "tempo_deslocamento_minutos": (
                        tempo_deslocamento_atendimento / 60
                    )
                })

            rotas_json.append({

                "dia": dia,

                "equipe": str(equipe_id),

                "metodo_otimizacao": metodo_otimizacao,

                "coords": rota_coords,

                "pontos": pontos,

                "distancia_km": round(
                    distancia_total,
                    2
                )
            })

            metricas.append({

                "dia": dia,

                "equipe": equipe_id,

                "metodo_otimizacao": metodo_otimizacao,

                "distancia_total_km": round(
                    distancia_total,
                    2
                ),

                "tempo_total_horas": round(
                    jornada_horas,
                    2
                ),

                "tempo_total_com_buffer_horas": round(
                    tempo_total_com_buffer / 3600,
                    2
                ),

                "tempo_atendimento_horas": round(
                    tempo_servicos / 3600,
                    2
                ),

                "tempo_deslocamento_horas": round(
                    tempo_deslocamento / 3600,
                    2
                ),

                "quantidade_servicos": len(df_eq),

                "rota_viavel": rota_viavel
            })

        # ------------------------------------------------------------
        # Comparacao pareada: balanceamento antigo (bin-packing) vs. novo
        # (espacial), ambos roteirizados so com ACO+2opt (ver secao 2 das
        # instrucoes). O lado espacial reaproveita os resultados "ACO + 2OPT"
        # ja calculados acima para as equipes de `equipes`; o lado
        # bin-packing precisa ser roteirizado agora, com o balanceamento
        # antigo (`alocar_equipes_bin_packing`).
        # v11: o comparativo com o bin-packing roda ISOLADO. Na v10, uma
        # falha de validacao do bin-packing (5 dias de 2019) interrompia o
        # dia inteiro, antes do registro da ablacao - era a causa do n = 359.
        # Agora a falha fica restrita a esta tabela auxiliar (registrada com
        # valores ausentes e a mensagem de erro), e o dia segue normalmente.
        distancia_espacial_total = acumulado_baselines_dia.get(
            "ACO + 2OPT", {}
        ).get("distancia_total_km", 0)

        jornada_excedida_espacial_rotas = acumulado_baselines_dia.get(
            "ACO + 2OPT", {}
        ).get("jornada_excedida_horas", 0)

        funcao_objetivo_espacial = calcular_funcao_objetivo(
            distancia_espacial_total,
            jornada_excedida_espacial_rotas,
            balanceamento_final["tempo_ocioso_total_horas"],
            balanceamento_final["diferenca_mais_menos_carregada_horas"]
        )

        jornadas_espacial_array = (
            np.array(jornadas_espacial_aco2opt)
            if jornadas_espacial_aco2opt
            else np.array([0])
        )

        registro_comparativo = {
            "dia": dia,
            "equipes_bin_packing": np.nan,
            "equipes_espacial": balanceamento_final["equipes_usadas"],
            "distancia_km_bin_packing": np.nan,
            "distancia_km_espacial": distancia_espacial_total,
            "jornada_media_h_bin_packing": np.nan,
            "jornada_media_h_espacial": jornadas_espacial_array.mean(),
            "jornada_dp_h_bin_packing": np.nan,
            "jornada_dp_h_espacial": jornadas_espacial_array.std(ddof=0),
            "ociosidade_media_h_bin_packing": np.nan,
            "ociosidade_media_h_espacial": balanceamento_final[
                "tempo_ocioso_medio_horas"
            ],
            "funcao_objetivo_bin_packing": np.nan,
            "funcao_objetivo_espacial": funcao_objetivo_espacial,
            "erro_bin_packing": ""
        }

        try:

            log_etapa(
                f"Dia {dia}: roteirizando (ACO+2opt) os clusters do "
                f"balanceamento bin-packing para comparativo"
            )

            equipes_bin_packing = alocar_equipes_bin_packing(df_dia)

            balanceamento_bin_packing = calcular_metricas_balanceamento(
                equipes_bin_packing
            )

            distancia_bin_packing_total = 0
            jornada_excedida_bin_packing_rotas = 0
            jornadas_bin_packing = []

            for indice_bp, equipe_bp in enumerate(equipes_bin_packing):

                df_eq_bp = pd.DataFrame(
                    equipe_bp["servicos"]
                ).reset_index(drop=True)

                coords_bp = list(
                    zip(
                        df_eq_bp["lat"],
                        df_eq_bp["lon"]
                    )
                )

                matriz_bp = criar_matriz(coords_bp)

                if len(coords_bp) <= 2:
                    path_bp = list(range(len(coords_bp)))
                else:
                    path_bp = melhorar_rota_2opt(
                        otimizar_rota_aco(
                            matriz_bp,
                            semente=semente_sensibilidade(
                                dia, indice_bp, "bin-packing", 0
                            )
                        ),
                        matriz_bp
                    )

                metrica_bp = avaliar_caminho(
                    path_bp,
                    df_eq_bp,
                    coords_bp,
                    matriz_bp
                )

                distancia_bin_packing_total += metrica_bp["distancia_total_km"]
                jornada_excedida_bin_packing_rotas += (
                    metrica_bp["jornada_excedida_horas"]
                )
                jornadas_bin_packing.append(metrica_bp["jornada_horas"])

            jornadas_bin_packing_array = (
                np.array(jornadas_bin_packing)
                if jornadas_bin_packing
                else np.array([0])
            )

            registro_comparativo.update({
                "equipes_bin_packing": balanceamento_bin_packing[
                    "equipes_usadas"
                ],
                "distancia_km_bin_packing": distancia_bin_packing_total,
                "jornada_media_h_bin_packing": jornadas_bin_packing_array.mean(),
                "jornada_dp_h_bin_packing": jornadas_bin_packing_array.std(ddof=0),
                "ociosidade_media_h_bin_packing": balanceamento_bin_packing[
                    "tempo_ocioso_medio_horas"
                ],
                "funcao_objetivo_bin_packing": calcular_funcao_objetivo(
                    distancia_bin_packing_total,
                    jornada_excedida_bin_packing_rotas,
                    balanceamento_bin_packing["tempo_ocioso_total_horas"],
                    balanceamento_bin_packing[
                        "diferenca_mais_menos_carregada_horas"
                    ]
                )
            })

            log_etapa(
                f"Dia {dia}: comparativo balanceamento - bin-packing "
                f"{distancia_bin_packing_total:.1f}km / espacial "
                f"{distancia_espacial_total:.1f}km"
            )

        except Exception as erro_bin_packing:

            registro_comparativo["erro_bin_packing"] = str(erro_bin_packing)

            log_etapa(
                f"Dia {dia}: comparativo bin-packing NAO calculado "
                f"({erro_bin_packing}) - o restante do dia segue normalmente"
            )

            registrar_erro_dia(
                dia,
                f"[apenas comparativo bin-packing] {erro_bin_packing}"
            )

        comparativo_balanceamento.append(registro_comparativo)

        referencia_fifo = acumulado_baselines_dia.get(
            "FIFO",
            {}
        )

        # Percorre TODOS os metodos de roteamento acumulados no dia (inclui
        # Simulated Annealing e OR-Tools, alem dos 4 originais) para montar
        # os totais diarios por metodo - usados em `resumo_baselines*.xlsx`
        # (ver `metricas_baselines_diario` abaixo). O estudo de ablacao
        # (`metricas_ablation`), por sua vez, so recebe os 4 cenarios
        # originais (balanceamento x metodo de roteamento) - ele NAO deve
        # ser alterado pela inclusao dos novos baselines.
        for metodo_baseline, totais in acumulado_baselines_dia.items():

            distancia_ref = referencia_fifo.get(
                "distancia_total_km",
                0
            )

            tempo_ref = referencia_fifo.get(
                "tempo_deslocamento_horas",
                0
            )

            reducao_distancia = (
                (distancia_ref - totais["distancia_total_km"])
                / distancia_ref
                * 100
                if distancia_ref > 0
                else 0
            )

            reducao_tempo = (
                (tempo_ref - totais["tempo_deslocamento_horas"])
                / tempo_ref
                * 100
                if tempo_ref > 0
                else 0
            )

            # v11: o termo J usa a jornada excedida medida nas ROTAS deste
            # metodo (e nao a estimativa Nearest Neighbor da alocacao, que
            # e sempre zero por construcao). Os termos O e B continuam
            # vindo da alocacao (carga planejada), comuns a todos os
            # metodos de roteamento do mesmo balanceamento.
            objetivo = calcular_funcao_objetivo(
                totais["distancia_total_km"],
                totais["jornada_excedida_horas"],
                balanceamento_final["tempo_ocioso_total_horas"],
                balanceamento_final["diferenca_mais_menos_carregada_horas"]
            )

            registro_diario_metodo = {
                "dia": dia,
                "cenario": f"Balanceamento + {metodo_baseline}",
                "metodo_rota": metodo_baseline,
                "distancia_total_km": totais["distancia_total_km"],
                "tempo_deslocamento_horas": totais[
                    "tempo_deslocamento_horas"
                ],
                "tempo_total_horas": totais["tempo_total_horas"],
                "quantidade_servicos": totais["quantidade_servicos"],
                "equipes_usadas": balanceamento_final["equipes_usadas"],
                "carga_media_horas": balanceamento_final["carga_media_horas"],
                "carga_desvio_padrao_horas": balanceamento_final[
                    "carga_desvio_padrao_horas"
                ],
                "coeficiente_variacao_carga": balanceamento_final[
                    "coeficiente_variacao_carga"
                ],
                "tempo_ocioso_total_horas": balanceamento_final[
                    "tempo_ocioso_total_horas"
                ],
                "diferenca_mais_menos_carregada_horas": balanceamento_final[
                    "diferenca_mais_menos_carregada_horas"
                ],
                # Jornada excedida nas rotas do metodo (termo J da FO).
                "jornada_excedida_total_horas": totais[
                    "jornada_excedida_horas"
                ],
                "rotas_inviaveis": totais["rotas_inviaveis"],
                "percentual_rotas_inviaveis": (
                    totais["rotas_inviaveis"]
                    / balanceamento_final["equipes_usadas"]
                    * 100
                    if balanceamento_final["equipes_usadas"] > 0
                    else 0
                ),
                # Estimativa da alocacao (Nearest Neighbor) - mantida para
                # transparencia; e sempre zero porque a alocacao so aceita
                # equipes que cabem na jornada segundo essa estimativa.
                "jornada_excedida_alocacao_horas": balanceamento_final[
                    "jornada_excedida_total_horas"
                ],
                "equipes_baixa_alocacao": balanceamento_final[
                    "equipes_baixa_alocacao"
                ],
                "percentual_equipes_baixa_alocacao": balanceamento_final[
                    "percentual_equipes_baixa_alocacao"
                ],
                "reducao_distancia_vs_fifo_percentual": reducao_distancia,
                "reducao_tempo_deslocamento_vs_fifo_percentual": reducao_tempo,
                "funcao_objetivo": objetivo
            }

            # Totais diarios por metodo (usados por resumo_baselines*.xlsx),
            # para TODOS os metodos.
            metricas_baselines_diario.append(registro_diario_metodo)

            # Estudo de ablacao (`metricas_ablation`/`ablation_study*.xlsx`):
            # somente os 4 cenarios originais, sem alteracao.
            if metodo_baseline in (
                "FIFO",
                "Nearest Neighbor",
                "ACO sem 2OPT",
                "ACO + 2OPT"
            ):
                metricas_ablation.append(registro_diario_metodo)

        objetivo_sem_balanceamento = calcular_funcao_objetivo(
            sem_balanceamento_fifo_dia["distancia_total_km"],
            sem_balanceamento_fifo_dia["jornada_excedida_horas"],
            balanceamento_fifo["tempo_ocioso_total_horas"],
            balanceamento_fifo[
                "diferenca_mais_menos_carregada_horas"
            ]
        )

        metricas_ablation.append({
            "dia": dia,
            "cenario": "Sem balanceamento + FIFO",
            "metodo_rota": "FIFO",
            "distancia_total_km": sem_balanceamento_fifo_dia[
                "distancia_total_km"
            ],
            "tempo_deslocamento_horas": sem_balanceamento_fifo_dia[
                "tempo_deslocamento_horas"
            ],
            "tempo_total_horas": sem_balanceamento_fifo_dia[
                "tempo_total_horas"
            ],
            "quantidade_servicos": sem_balanceamento_fifo_dia[
                "quantidade_servicos"
            ],
            "equipes_usadas": balanceamento_fifo["equipes_usadas"],
            "carga_media_horas": balanceamento_fifo["carga_media_horas"],
            "carga_desvio_padrao_horas": balanceamento_fifo[
                "carga_desvio_padrao_horas"
            ],
            "coeficiente_variacao_carga": balanceamento_fifo[
                "coeficiente_variacao_carga"
            ],
            "tempo_ocioso_total_horas": balanceamento_fifo[
                "tempo_ocioso_total_horas"
            ],
            "diferenca_mais_menos_carregada_horas": balanceamento_fifo[
                "diferenca_mais_menos_carregada_horas"
            ],
            "jornada_excedida_total_horas": sem_balanceamento_fifo_dia[
                "jornada_excedida_horas"
            ],
            "rotas_inviaveis": sem_balanceamento_fifo_dia["rotas_inviaveis"],
            "percentual_rotas_inviaveis": (
                sem_balanceamento_fifo_dia["rotas_inviaveis"]
                / balanceamento_fifo["equipes_usadas"]
                * 100
                if balanceamento_fifo["equipes_usadas"] > 0
                else 0
            ),
            "jornada_excedida_alocacao_horas": balanceamento_fifo[
                "jornada_excedida_total_horas"
            ],
            "equipes_baixa_alocacao": balanceamento_fifo[
                "equipes_baixa_alocacao"
            ],
            "percentual_equipes_baixa_alocacao": balanceamento_fifo[
                "percentual_equipes_baixa_alocacao"
            ],
            "reducao_distancia_vs_fifo_percentual": 0,
            "reducao_tempo_deslocamento_vs_fifo_percentual": 0,
            "funcao_objetivo": objetivo_sem_balanceamento
        })

        log_etapa(
            f"Dia {dia} concluido em {time.perf_counter() - inicio_dia:.1f}s"
        )


        _dados_checkpoint = {
            "rotas_json": rotas_json[_snap_rotas_json:],
            "metricas": metricas[_snap_metricas:],
            "sequencias": sequencias[_snap_sequencias:],
            "metricas_baselines": metricas_baselines[_snap_metricas_baselines:],
            "metricas_balanceamento": metricas_balanceamento[
                _snap_metricas_balanceamento:
            ],
            "metricas_ablation": metricas_ablation[_snap_metricas_ablation:],
            "metricas_baselines_diario": metricas_baselines_diario[
                _snap_metricas_baselines_diario:
            ],
            "cargas_equipes_boxplot": cargas_equipes_boxplot[
                _snap_cargas_equipes_boxplot:
            ],
            "comparativo_2opt": comparativo_2opt[_snap_comparativo_2opt:],
            "comparativo_balanceamento": comparativo_balanceamento[
                _snap_comparativo_balanceamento:
            ],
            "registros_replicacoes_aco_pipeline": (
                REGISTROS_REPLICACOES_ACO_PIPELINE[_snap_replicacoes_aco:]
            )
        }

        with open(_checkpoint_path, "wb") as _f_checkpoint:
            pickle.dump(_dados_checkpoint, _f_checkpoint)

    except Exception as _erro_dia:

        log_etapa(f"ERRO ao processar dia {dia}: {_erro_dia}")

        # v11: reverte TODAS as listas globais ao estado anterior ao dia.
        # Na v10, os registros ja acrescentados antes do erro permaneciam
        # em algumas tabelas (ex.: metricas_baselines) e nao em outras
        # (ex.: metricas_ablation), o que gerava n diferentes entre tabelas.
        del rotas_json[_snap_rotas_json:]
        del metricas[_snap_metricas:]
        del sequencias[_snap_sequencias:]
        del metricas_baselines[_snap_metricas_baselines:]
        del metricas_balanceamento[_snap_metricas_balanceamento:]
        del metricas_ablation[_snap_metricas_ablation:]
        del metricas_baselines_diario[_snap_metricas_baselines_diario:]
        del cargas_equipes_boxplot[_snap_cargas_equipes_boxplot:]
        del comparativo_2opt[_snap_comparativo_2opt:]
        del comparativo_balanceamento[_snap_comparativo_balanceamento:]
        del REGISTROS_REPLICACOES_ACO_PIPELINE[_snap_replicacoes_aco:]

        registrar_erro_dia(dia, f"[dia inteiro descartado] {_erro_dia}")

        continue
# ==========================================================
# EXPORTAÇÃO
# ==========================================================
# A partir daqui, as listas acumuladas durante o loop diário são
# convertidas em DataFrames e agregadas em resumos estatísticos
# (média ± IC95 por cenário/dia), que alimentam todas as tabelas e
# figuras do TCC. Cada `to_excel(...)` mais adiante corresponde a um
# arquivo específico citado nas fontes das tabelas/figuras do
# documento de resultados (ex.: resumo_ablation_ic95.xlsx,
# metricas_balanceamento.xlsx etc.).

print("\nEXPORTANDO EXCEL...")

inicio = time.perf_counter()

df_metricas = pd.DataFrame(metricas)

df_baselines = pd.DataFrame(metricas_baselines)

df_balanceamento = pd.DataFrame(metricas_balanceamento)

# ==========================================================
# RESUMO ESTATISTICO DO BALANCEAMENTO
# ==========================================================

resumo_balanceamento = pd.DataFrame()

if len(df_balanceamento) > 0:

    resumo_balanceamento_completo = resumir_por_metodo(
        df_balanceamento,
        "cenario"
    )

    metricas_resumo_balanceamento = [
        "equipes_usadas",
        "carga_media_horas",
        "carga_desvio_padrao_horas",
        "coeficiente_variacao_carga",
        "tempo_ocioso_total_horas",
        "diferenca_mais_menos_carregada_horas"
    ]

    colunas_resumo_balanceamento = ["cenario", "dias"]

    for metrica in metricas_resumo_balanceamento:

        colunas_resumo_balanceamento += [
            f"{metrica}_media",
            f"{metrica}_desvio_padrao",
            f"{metrica}_ic95"
        ]

    resumo_balanceamento = resumo_balanceamento_completo[
        [
            coluna
            for coluna in colunas_resumo_balanceamento
            if coluna in resumo_balanceamento_completo.columns
        ]
    ]

df_ablation = pd.DataFrame(metricas_ablation)

# Totais diarios por metodo, para TODOS os metodos de roteamento
# (inclui Simulated Annealing e OR-Tools) - fonte de
# `resumo_baselines*.xlsx`, ver `metricas_baselines_diario` acima.
df_baselines_diario = pd.DataFrame(metricas_baselines_diario)

# ==========================================================
# TESTES ESTATISTICOS DE SIGNIFICANCIA
# (teste t pareado e teste de Wilcoxon, com correcao de Holm-Bonferroni)
# ==========================================================
#
# Compara pares de cenarios executados sobre os mesmos dias de
# simulacao para verificar se os ganhos observados no estudo de
# ablacao (balanceamento, ACO e 2OPT), nos indicadores de
# balanceamento de carga e na comparacao do ACO + 2OPT com os baselines
# competitivos (Simulated Annealing e OR-Tools) sao estatisticamente
# significativos. Os p-valores de cada teste sao ajustados pelo metodo
# de Holm-Bonferroni sobre o conjunto de TODAS as comparacoes da tabela
# (colunas p_holm_t e p_holm_wilcoxon).
#
# Sao aplicados dois testes complementares:
#   - teste t pareado (scipy.stats.ttest_rel): assume distribuicao
#     aproximadamente normal das diferencas diarias;
#   - teste de Wilcoxon (signed-rank; scipy.stats.wilcoxon): nao
#     parametrico, mais robusto a assimetrias e valores atipicos,
#     relevante pois o numero de equipes disponiveis varia entre os
#     dias simulados.
#
# NIVEL_SIGNIFICANCIA define o alpha adotado (5%) para a coluna de
# interpretacao.

NIVEL_SIGNIFICANCIA = 0.05


def calcular_teste_pareado(x, y):
    """Calcula teste t pareado e teste de Wilcoxon entre duas
    series pareadas (mesmos dias), retornando None se nao for
    possivel calcular (ex.: series identicas ou poucos pares)."""

    diferenca = x - y

    resultado = {
        "n_pares": len(diferenca),
        "diferenca_media": diferenca.mean(),
        "diferenca_desvio_padrao": diferenca.std(ddof=1),
        "estatistica_t": np.nan,
        "p_valor_t": np.nan,
        "estatistica_wilcoxon": np.nan,
        "p_valor_wilcoxon": np.nan
    }

    try:

        estatistica_t, p_valor_t = ttest_rel(x, y)

        resultado["estatistica_t"] = estatistica_t
        resultado["p_valor_t"] = p_valor_t

    except ValueError as erro:

        log_etapa(f"Teste t pareado nao calculado: {erro}")

    try:

        estatistica_w, p_valor_w = wilcoxon(x, y)

        resultado["estatistica_wilcoxon"] = estatistica_w
        resultado["p_valor_wilcoxon"] = p_valor_w

    except ValueError as erro:

        log_etapa(f"Teste de Wilcoxon nao calculado: {erro}")

    return resultado


def ajustar_p_holm(p_valores):
    """Correcao de Holm-Bonferroni (Holm, 1979) para comparacoes
    multiplas: ordena os m p-valores em ordem crescente, multiplica o
    i-esimo (i = 0, 1, ...) por (m - i), impoe monotonicidade (cada
    p ajustado >= o anterior) e limita a 1. Valores ausentes (NaN) sao
    ignorados e permanecem NaN. Equivale a
    statsmodels.stats.multitest.multipletests(..., method="holm")."""

    p_valores = pd.Series(p_valores, dtype=float)
    validos = p_valores.dropna()
    ajustados = pd.Series(np.nan, index=p_valores.index)

    if len(validos) == 0:
        return ajustados

    ordem = validos.sort_values(kind="stable")
    m = len(ordem)
    maximo_acumulado = 0.0

    for posicao, (indice, p_valor) in enumerate(ordem.items()):
        valor = min(1.0, (m - posicao) * p_valor)
        maximo_acumulado = max(maximo_acumulado, valor)
        ajustados[indice] = maximo_acumulado

    return ajustados


registros_testes_significancia = []

# --- Comparacoes do estudo de ablacao (distancia_total_km) -------

comparacoes_ablation_significancia = [
    (
        "Balanceamento (Sem vs. Com)",
        "Sem balanceamento + FIFO",
        "Balanceamento + FIFO",
        "distancia_total_km"
    ),
    (
        "ACO (Balanc. + FIFO -> Balanc. + ACO sem 2OPT)",
        "Balanceamento + FIFO",
        "Balanceamento + ACO sem 2OPT",
        "distancia_total_km"
    ),
    (
        "2OPT (Balanc. + ACO sem 2OPT -> Balanc. + ACO + 2OPT)",
        "Balanceamento + ACO sem 2OPT",
        "Balanceamento + ACO + 2OPT",
        "distancia_total_km"
    ),
    (
        "Modelo completo vs. referencia "
        "(Sem balanc. + FIFO -> Balanc. + ACO + 2OPT)",
        "Sem balanceamento + FIFO",
        "Balanceamento + ACO + 2OPT",
        "distancia_total_km"
    )
]

if len(df_ablation) > 0:

    for rotulo, cenario_a, cenario_b, metrica in (
        comparacoes_ablation_significancia
    ):

        pivot_metrica = df_ablation[
            df_ablation["cenario"].isin([cenario_a, cenario_b])
        ].pivot(
            index="dia",
            columns="cenario",
            values=metrica
        ).dropna()

        if (
            cenario_a in pivot_metrica.columns
            and
            cenario_b in pivot_metrica.columns
            and
            len(pivot_metrica) > 0
        ):

            resultado = calcular_teste_pareado(
                pivot_metrica[cenario_a],
                pivot_metrica[cenario_b]
            )

            resultado["comparacao"] = rotulo
            resultado["metrica"] = metrica

            registros_testes_significancia.append(resultado)

# --- ACO + 2OPT vs. baselines competitivos (mesmo balanceamento) -----
# Diferenca = ACO + 2OPT - metodo concorrente: valores POSITIVOS indicam
# que o concorrente percorreu MENOS distancia que o ACO + 2OPT.
if (
    len(df_baselines_diario) > 0
    and
    "metodo_rota" in df_baselines_diario.columns
):
    pivot_baselines = df_baselines_diario.pivot(
        index="dia",
        columns="metodo_rota",
        values="distancia_total_km"
    )
    for metodo_concorrente in ("Simulated Annealing", "OR-Tools"):
        if (
            "ACO + 2OPT" in pivot_baselines.columns
            and
            metodo_concorrente in pivot_baselines.columns
        ):
            pares = pivot_baselines[
                ["ACO + 2OPT", metodo_concorrente]
            ].dropna()
            if len(pares) > 0:
                resultado = calcular_teste_pareado(
                    pares["ACO + 2OPT"],
                    pares[metodo_concorrente]
                )
                resultado["comparacao"] = (
                    f"ACO + 2OPT vs. {metodo_concorrente} "
                    f"(ambos sob balanceamento espacial)"
                )
                resultado["metrica"] = "distancia_total_km"
                resultado["percentual_dias_concorrente_melhor"] = (
                    (pares[metodo_concorrente] < pares["ACO + 2OPT"]).mean()
                    * 100
                )
                registros_testes_significancia.append(resultado)

# --- Indicadores de balanceamento de carga (Sem vs. Com) ---------

metricas_balanceamento_significancia = [
    "carga_desvio_padrao_horas",
    "coeficiente_variacao_carga",
    "tempo_ocioso_total_horas",
    "diferenca_mais_menos_carregada_horas",
    "equipes_usadas"
]

if len(df_balanceamento) > 0:

    for metrica in metricas_balanceamento_significancia:

        pivot_metrica = df_balanceamento.pivot(
            index="dia",
            columns="cenario",
            values=metrica
        ).dropna()

        if (
            "Sem balanceamento" in pivot_metrica.columns
            and
            "Com balanceamento" in pivot_metrica.columns
            and
            len(pivot_metrica) > 0
        ):

            resultado = calcular_teste_pareado(
                pivot_metrica["Sem balanceamento"],
                pivot_metrica["Com balanceamento"]
            )

            resultado["comparacao"] = "Balanceamento (Sem vs. Com)"
            resultado["metrica"] = metrica

            registros_testes_significancia.append(resultado)

teste_significancia_estatistica = pd.DataFrame(
    registros_testes_significancia
)

if len(teste_significancia_estatistica) > 0:

    colunas_ordenadas = [
        "comparacao",
        "metrica",
        "n_pares",
        "diferenca_media",
        "diferenca_desvio_padrao",
        "estatistica_t",
        "p_valor_t",
        "estatistica_wilcoxon",
        "p_valor_wilcoxon"
    ]

    teste_significancia_estatistica["p_holm_t"] = ajustar_p_holm(
        teste_significancia_estatistica["p_valor_t"]
    )
    teste_significancia_estatistica["p_holm_wilcoxon"] = ajustar_p_holm(
        teste_significancia_estatistica["p_valor_wilcoxon"]
    )

    # d de Cohen pareado: diferenca media / desvio padrao das diferencas.
    teste_significancia_estatistica["d_cohen_pareado"] = (
        teste_significancia_estatistica["diferenca_media"]
        /
        teste_significancia_estatistica["diferenca_desvio_padrao"]
    )

    colunas_ordenadas += [
        "p_holm_t",
        "p_holm_wilcoxon",
        "d_cohen_pareado"
    ]

    if "percentual_dias_concorrente_melhor" in (
        teste_significancia_estatistica.columns
    ):
        colunas_ordenadas.append("percentual_dias_concorrente_melhor")

    teste_significancia_estatistica = teste_significancia_estatistica[
        colunas_ordenadas
    ]

    def interpretar_significancia(p_t, p_w):

        if pd.isna(p_t) or pd.isna(p_w):
            return "Nao calculado"

        significativo_t = p_t < NIVEL_SIGNIFICANCIA
        significativo_w = p_w < NIVEL_SIGNIFICANCIA

        if significativo_t and significativo_w:
            return "Significativo em ambos os testes (p Holm < 0.05)"

        if not significativo_t and not significativo_w:
            return "Sem diferenca estatisticamente significativa (p Holm >= 0.05)"

        return (
            "Resultado inconsistente entre os testes "
            "(significativo em apenas um deles)"
        )

    teste_significancia_estatistica["interpretacao"] = (
        teste_significancia_estatistica.apply(
            lambda linha: interpretar_significancia(
                linha["p_holm_t"],
                linha["p_holm_wilcoxon"]
            ),
            axis=1
        )
    )

# Mantido por compatibilidade com o restante do script/relatorios
# anteriores, que exportam a variavel "teste_wilcoxon".
teste_wilcoxon = teste_significancia_estatistica

df_cargas_boxplot = pd.DataFrame(cargas_equipes_boxplot)

df_2opt = pd.DataFrame(comparativo_2opt)

# v11: a winsorizacao P5-P95 da v10 foi removida. Ela nao alterava os
# quartis do boxplot (apenas os extremos) e a figura correspondente
# ocultava os outliers; a nova figura (ver secao de graficos) usa os
# valores originais, com os outliers visiveis.

if (
    len(df_baselines_diario) > 0
    and
    "metodo_rota" in df_baselines_diario.columns
):

    resumo_baselines = resumir_por_metodo(
        df_baselines_diario,
        "metodo_rota"
    ).rename(
        columns={
            "metodo_rota": "metodo",
            "distancia_total_km_desvio_padrao": (
                "distancia_total_km_desvio"
            ),
            "tempo_deslocamento_horas_desvio_padrao": (
                "tempo_deslocamento_horas_desvio"
            )
        }
    )

    colunas_resumo_baselines = [
        "metodo",
        "distancia_total_km_media",
        "distancia_total_km_desvio",
        "distancia_total_km_ic95",
        "tempo_deslocamento_horas_media",
        "tempo_deslocamento_horas_desvio",
        "tempo_deslocamento_horas_ic95",
        "equipes_usadas_media"
    ]

    resumo_baselines = resumo_baselines[
        [
            coluna
            for coluna in colunas_resumo_baselines
            if coluna in resumo_baselines.columns
        ]
    ]

    df_resumo_baselines = resumo_baselines

    df_resumo_ablation = resumir_por_metodo(
        df_ablation,
        "cenario"
    )

else:

    resumo_baselines = pd.DataFrame()

    df_resumo_baselines = resumo_baselines

    df_resumo_ablation = pd.DataFrame()

# ==========================================================
# COMPARATIVO DE GANHOS (FIFO COMO BASELINE)
# ==========================================================

comparativo_ganhos = pd.DataFrame()

if (
    len(df_resumo_baselines) > 0
    and
    "metodo" in df_resumo_baselines.columns
):

    referencia_fifo_baseline = df_resumo_baselines.loc[
        df_resumo_baselines["metodo"] == "FIFO"
    ]

    if len(referencia_fifo_baseline) > 0:

        distancia_fifo = referencia_fifo_baseline[
            "distancia_total_km_media"
        ].iloc[0]

        tempo_fifo = referencia_fifo_baseline[
            "tempo_deslocamento_horas_media"
        ].iloc[0]

        registros_comparativo_ganhos = []

        for _, linha in df_resumo_baselines.iterrows():

            distancia_metodo = linha["distancia_total_km_media"]

            tempo_metodo = linha["tempo_deslocamento_horas_media"]

            ganho_distancia_percentual = (
                (distancia_fifo - distancia_metodo)
                / distancia_fifo
                * 100
                if distancia_fifo > 0
                else 0
            )

            ganho_tempo_deslocamento_percentual = (
                (tempo_fifo - tempo_metodo)
                / tempo_fifo
                * 100
                if tempo_fifo > 0
                else 0
            )

            registros_comparativo_ganhos.append({
                "metodo": linha["metodo"],
                "ganho_distancia_percentual": ganho_distancia_percentual,
                "ganho_tempo_deslocamento_percentual": (
                    ganho_tempo_deslocamento_percentual
                )
            })

        comparativo_ganhos = pd.DataFrame(registros_comparativo_ganhos)

# ==========================================================
# TABELA DE ABLATION STUDY
# ==========================================================

ablation_study = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenarios_ablation_study = [
        "Sem balanceamento + FIFO",
        "Balanceamento + Nearest Neighbor",
        "Balanceamento + ACO sem 2OPT",
        "Balanceamento + ACO + 2OPT"
    ]

    ablation_study = df_resumo_ablation[
        df_resumo_ablation["cenario"].isin(cenarios_ablation_study)
    ][
        [
            "cenario",
            "funcao_objetivo_media",
            "distancia_total_km_media",
            "tempo_deslocamento_horas_media",
            "equipes_usadas_media"
        ]
    ].copy()

    ablation_study["cenario"] = pd.Categorical(
        ablation_study["cenario"],
        categories=cenarios_ablation_study,
        ordered=True
    )

    ablation_study = ablation_study.sort_values(
        "cenario"
    ).reset_index(drop=True)

    referencia_ablation_study = ablation_study.loc[
        ablation_study["cenario"] == "Sem balanceamento + FIFO"
    ]

    if len(referencia_ablation_study) > 0:

        objetivo_baseline_ablation_study = referencia_ablation_study[
            "funcao_objetivo_media"
        ].iloc[0]

        ablation_study["ganho_percentual_funcao_objetivo"] = (
            (
                objetivo_baseline_ablation_study
                -
                ablation_study["funcao_objetivo_media"]
            )
            /
            objetivo_baseline_ablation_study
            *
            100
            if objetivo_baseline_ablation_study > 0
            else 0
        )

# ==========================================================
# EXPERIMENTO: EFEITO ISOLADO DO BALANCEAMENTO (FIFO)
# ==========================================================

ganho_balanceamento = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenario_sem_balanceamento = "Sem balanceamento + FIFO"

    cenario_com_balanceamento = "Balanceamento + FIFO"

    referencia_sem = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_sem_balanceamento
    ]

    referencia_com = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_com_balanceamento
    ]

    if (
        len(referencia_sem) > 0
        and
        len(referencia_com) > 0
    ):

        metricas_ganho_balanceamento = {
            "distancia_total_km": "distancia_total_km_media",
            "equipes_usadas": "equipes_usadas_media",
            "carga_media_horas": "carga_media_horas_media",
            "carga_desvio_padrao_horas": "carga_desvio_padrao_horas_media",
            "coeficiente_variacao_carga": "coeficiente_variacao_carga_media",
            "tempo_ocioso_total_horas": "tempo_ocioso_total_horas_media"
        }

        registros_ganho_balanceamento = []

        for metrica, coluna_media in metricas_ganho_balanceamento.items():

            valor_sem = referencia_sem[coluna_media].iloc[0]

            valor_com = referencia_com[coluna_media].iloc[0]

            ganho_percentual = (
                (valor_sem - valor_com)
                / valor_sem
                * 100
                if valor_sem > 0
                else 0
            )

            registros_ganho_balanceamento.append({
                "metrica": metrica,
                "sem_balanceamento_fifo": valor_sem,
                "com_balanceamento_fifo": valor_com,
                "ganho_percentual": ganho_percentual
            })

        ganho_balanceamento = pd.DataFrame(
            registros_ganho_balanceamento
        )

# ==========================================================
# EXPERIMENTO: GANHO ESPECIFICO DO ACO
# ==========================================================

ganho_aco = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenario_nearest_neighbor = "Balanceamento + Nearest Neighbor"

    cenario_aco_sem_2opt = "Balanceamento + ACO sem 2OPT"

    referencia_nn = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_nearest_neighbor
    ]

    referencia_aco = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_aco_sem_2opt
    ]

    if (
        len(referencia_nn) > 0
        and
        len(referencia_aco) > 0
    ):

        metricas_ganho_aco = {
            "distancia_total_km": "distancia_total_km_media",
            "tempo_deslocamento_horas": "tempo_deslocamento_horas_media",
            "funcao_objetivo": "funcao_objetivo_media"
        }

        registros_ganho_aco = []

        for metrica, coluna_media in metricas_ganho_aco.items():

            valor_nn = referencia_nn[coluna_media].iloc[0]

            valor_aco = referencia_aco[coluna_media].iloc[0]

            ganho_percentual = (
                (valor_nn - valor_aco)
                / valor_nn
                * 100
                if valor_nn > 0
                else 0
            )

            registros_ganho_aco.append({
                "metrica": metrica,
                "balanceamento_nearest_neighbor": valor_nn,
                "balanceamento_aco_sem_2opt": valor_aco,
                "ganho_percentual": ganho_percentual
            })

        ganho_aco = pd.DataFrame(registros_ganho_aco)

# ==========================================================
# EXPERIMENTO: GANHO ESPECIFICO DO 2OPT
# ==========================================================

ganho_2opt_experimento = pd.DataFrame()

if (
    len(df_resumo_ablation) > 0
    and
    "cenario" in df_resumo_ablation.columns
):

    cenario_aco_sem_2opt = "Balanceamento + ACO sem 2OPT"

    cenario_aco_2opt = "Balanceamento + ACO + 2OPT"

    referencia_aco_sem_2opt = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_aco_sem_2opt
    ]

    referencia_aco_2opt = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == cenario_aco_2opt
    ]

    if (
        len(referencia_aco_sem_2opt) > 0
        and
        len(referencia_aco_2opt) > 0
    ):

        metricas_ganho_2opt_experimento = {
            "distancia_total_km": "distancia_total_km_media",
            "tempo_deslocamento_horas": "tempo_deslocamento_horas_media",
            "funcao_objetivo": "funcao_objetivo_media"
        }

        registros_ganho_2opt_experimento = []

        for metrica, coluna_media in metricas_ganho_2opt_experimento.items():

            valor_aco_sem_2opt = referencia_aco_sem_2opt[
                coluna_media
            ].iloc[0]

            valor_aco_2opt = referencia_aco_2opt[coluna_media].iloc[0]

            ganho_percentual = (
                (valor_aco_sem_2opt - valor_aco_2opt)
                / valor_aco_sem_2opt
                * 100
                if valor_aco_sem_2opt > 0
                else 0
            )

            registros_ganho_2opt_experimento.append({
                "metrica": metrica,
                "balanceamento_aco_sem_2opt": valor_aco_sem_2opt,
                "balanceamento_aco_2opt": valor_aco_2opt,
                "ganho_percentual": ganho_percentual
            })

        ganho_2opt_experimento = pd.DataFrame(
            registros_ganho_2opt_experimento
        )

# ==========================================================
# TABELA FINAL CONSOLIDADA PARA PUBLICACAO
# ==========================================================

tabela_artigo = pd.DataFrame()

if (
    len(df_resumo_baselines) > 0
    and
    len(comparativo_ganhos) > 0
    and
    "metodo" in df_resumo_baselines.columns
    and
    "metodo" in comparativo_ganhos.columns
):

    ordem_metodos_artigo = [
        "FIFO",
        "Nearest Neighbor",
        "ACO sem 2OPT",
        "ACO + 2OPT"
    ]

    tabela_artigo = df_resumo_baselines.merge(
        comparativo_ganhos,
        on="metodo",
        how="inner"
    )

    tabela_artigo = tabela_artigo[
        tabela_artigo["metodo"].isin(ordem_metodos_artigo)
    ]

    tabela_artigo["metodo"] = pd.Categorical(
        tabela_artigo["metodo"],
        categories=ordem_metodos_artigo,
        ordered=True
    )

    tabela_artigo = tabela_artigo.sort_values(
        "metodo"
    ).reset_index(drop=True)

    tabela_artigo = tabela_artigo.rename(columns={
        "metodo": "Método",
        "distancia_total_km_media": "Distância Média (km)",
        "distancia_total_km_ic95": "IC95 Distância",
        "tempo_deslocamento_horas_media": "Tempo Médio Deslocamento (h)",
        "tempo_deslocamento_horas_ic95": "IC95 Tempo",
        "ganho_distancia_percentual": "Ganho Distância (%)",
        "ganho_tempo_deslocamento_percentual": "Ganho Tempo (%)"
    })

    tabela_artigo = tabela_artigo[
        [
            "Método",
            "Distância Média (km)",
            "IC95 Distância",
            "Tempo Médio Deslocamento (h)",
            "IC95 Tempo",
            "Ganho Distância (%)",
            "Ganho Tempo (%)"
        ]
    ]

    colunas_numericas_artigo = [
        coluna
        for coluna in tabela_artigo.columns
        if coluna != "Método"
    ]

    tabela_artigo[colunas_numericas_artigo] = tabela_artigo[
        colunas_numericas_artigo
    ].round(2)

if len(df_resumo_ablation) > 0:

    referencia_ablation = df_resumo_ablation.loc[
        df_resumo_ablation["cenario"] == "Sem balanceamento + FIFO"
    ]

    if len(referencia_ablation) > 0:

        distancia_base = referencia_ablation[
            "distancia_total_km_media"
        ].iloc[0]

        tempo_base = referencia_ablation[
            "tempo_deslocamento_horas_media"
        ].iloc[0]

        equipes_base = referencia_ablation[
            "equipes_usadas_media"
        ].iloc[0]

        carga_base = referencia_ablation[
            "carga_media_horas_media"
        ].iloc[0]

        df_resumo_ablation[
            "ganho_distancia_percentual"
        ] = (
            (distancia_base - df_resumo_ablation[
                "distancia_total_km_media"
            ])
            /
            distancia_base
            *
            100
            if distancia_base > 0
            else 0
        )

        df_resumo_ablation[
            "ganho_tempo_deslocamento_percentual"
        ] = (
            (tempo_base - df_resumo_ablation[
                "tempo_deslocamento_horas_media"
            ])
            /
            tempo_base
            *
            100
            if tempo_base > 0
            else 0
        )

        df_resumo_ablation[
            "reducao_uso_equipes_percentual"
        ] = (
            (equipes_base - df_resumo_ablation[
                "equipes_usadas_media"
            ])
            /
            equipes_base
            *
            100
            if equipes_base > 0
            else 0
        )

        df_resumo_ablation[
            "ganho_carga_media_percentual"
        ] = (
            (df_resumo_ablation[
                "carga_media_horas_media"
            ] - carga_base)
            /
            carga_base
            *
            100
            if carga_base > 0
            else 0
        )

df_ganhos_ablation = pd.DataFrame()

if len(df_ablation) > 0:

    cenarios_ganho = {
        "ganho_balanceamento": (
            "Sem balanceamento + FIFO",
            "Balanceamento + FIFO"
        ),
        "ganho_aco": (
            "Balanceamento + FIFO",
            "Balanceamento + ACO sem 2OPT"
        ),
        "ganho_2opt": (
            "Balanceamento + ACO sem 2OPT",
            "Balanceamento + ACO + 2OPT"
        )
    }

    registros_ganhos = []

    for dia_ganho, grupo_dia in df_ablation.groupby("dia"):

        grupo_idx = grupo_dia.set_index("cenario")

        for componente, (antes, depois) in cenarios_ganho.items():

            if (
                antes not in grupo_idx.index
                or
                depois not in grupo_idx.index
            ):
                continue

            linha_antes = grupo_idx.loc[antes]
            linha_depois = grupo_idx.loc[depois]

            registros_ganhos.append({
                "dia": dia_ganho,
                "componente": componente,
                "cenario_antes": antes,
                "cenario_depois": depois,
                "ganho_distancia_percentual": (
                    (
                        linha_antes["distancia_total_km"]
                        -
                        linha_depois["distancia_total_km"]
                    )
                    /
                    linha_antes["distancia_total_km"]
                    *
                    100
                    if linha_antes["distancia_total_km"] > 0
                    else 0
                ),
                "ganho_tempo_deslocamento_percentual": (
                    (
                        linha_antes["tempo_deslocamento_horas"]
                        -
                        linha_depois["tempo_deslocamento_horas"]
                    )
                    /
                    linha_antes["tempo_deslocamento_horas"]
                    *
                    100
                    if linha_antes["tempo_deslocamento_horas"] > 0
                    else 0
                ),
                "ganho_funcao_objetivo_percentual": (
                    (
                        linha_antes["funcao_objetivo"]
                        -
                        linha_depois["funcao_objetivo"]
                    )
                    /
                    linha_antes["funcao_objetivo"]
                    *
                    100
                    if linha_antes["funcao_objetivo"] > 0
                    else 0
                ),
                "variacao_equipes_percentual": (
                    (
                        linha_antes["equipes_usadas"]
                        -
                        linha_depois["equipes_usadas"]
                    )
                    /
                    linha_antes["equipes_usadas"]
                    *
                    100
                    if linha_antes["equipes_usadas"] > 0
                    else 0
                ),
                "variacao_carga_media_percentual": (
                    (
                        linha_depois["carga_media_horas"]
                        -
                        linha_antes["carga_media_horas"]
                    )
                    /
                    linha_antes["carga_media_horas"]
                    *
                    100
                    if linha_antes["carga_media_horas"] > 0
                    else 0
                )
            })

    df_ganhos_ablation = pd.DataFrame(registros_ganhos)

df_resumo_ganhos_ablation = resumir_por_metodo(
    df_ganhos_ablation.rename(
        columns={
            "ganho_distancia_percentual": "distancia_total_km",
            "ganho_tempo_deslocamento_percentual": (
                "tempo_deslocamento_horas"
            ),
            "ganho_funcao_objetivo_percentual": "funcao_objetivo",
            "variacao_equipes_percentual": "equipes_usadas",
            "variacao_carga_media_percentual": "carga_media_horas"
        }
    ),
    "componente"
) if len(df_ganhos_ablation) > 0 else pd.DataFrame()

if len(resumo_baselines) > 0:

    print("\nRESUMO BASELINES:")

    print(
        resumo_baselines
        .sort_values("distancia_total_km_media")
        .to_string(index=False)
    )

df_metricas.to_excel(
    caminho_tabela("metricas_operacionais.xlsx"),
    index=False
)

pd.DataFrame(sequencias).to_excel(
    caminho_tabela("sequencia_atendimentos.xlsx"),
    index=False
)

df_baselines.to_excel(
    caminho_tabela("metricas_baselines_rotas.xlsx"),
    index=False
)

# v11: tempo de execucao por metodo de roteamento e faixa de tamanho da
# rota (numero de pontos) - base para a discussao de custo computacional
# e escalabilidade. Inclui o custo de montar a matriz de distancias.
if (
    len(df_baselines) > 0
    and
    "tempo_execucao_segundos" in df_baselines.columns
):
    df_escalabilidade = df_baselines[
        df_baselines["quantidade_servicos"] >= 2
    ].copy()

    df_escalabilidade["faixa_pontos"] = pd.cut(
        df_escalabilidade["quantidade_servicos"],
        bins=[0, 10, 20, 30, 40, 1000],
        labels=["ate 10", "11 a 20", "21 a 30", "31 a 40", "acima de 40"]
    )

    df_escalabilidade.groupby(
        ["metodo", "faixa_pontos"],
        observed=True
    ).agg(
        rotas=("tempo_execucao_segundos", "size"),
        pontos_medio=("quantidade_servicos", "mean"),
        tempo_medio_segundos=("tempo_execucao_segundos", "mean"),
        tempo_mediano_segundos=("tempo_execucao_segundos", "median"),
        tempo_p90_segundos=(
            "tempo_execucao_segundos",
            lambda serie: serie.quantile(0.9)
        ),
        tempo_maximo_segundos=("tempo_execucao_segundos", "max")
    ).reset_index().to_excel(
        caminho_tabela("escalabilidade_metodos.xlsx"),
        index=False
    )

df_balanceamento.to_excel(
    caminho_tabela("metricas_balanceamento.xlsx"),
    index=False
)

# ==========================================================
# COMPARATIVO PAREADO: BALANCEAMENTO BIN-PACKING vs. ESPACIAL
# ==========================================================
# Ver secao 2 das instrucoes de alteracao: compara, dia a dia, o
# balanceamento antigo (`alocar_equipes_bin_packing`) com o novo
# balanceamento espacial (`alocar_equipes_balanceado`), ambos roteirizados
# apenas com ACO+2opt - isola o efeito do AGRUPAMENTO geografico do efeito
# da ORDEM de visita (que e papel do ACO/2opt, nao alterado).

df_comparativo_balanceamento = pd.DataFrame(comparativo_balanceamento)

df_comparativo_balanceamento.to_excel(
    caminho_tabela("comparativo_balanceamento.xlsx"),
    index=False
)

if len(df_comparativo_balanceamento) > 0:

    metricas_comparativo = [
        "equipes",
        "distancia_km",
        "jornada_media_h",
        "jornada_dp_h",
        "ociosidade_media_h",
        "funcao_objetivo"
    ]

    linhas_resumo_comparativo = []

    for metrica in metricas_comparativo:

        col_bin = f"{metrica}_bin_packing"
        col_esp = f"{metrica}_espacial"

        valores_bin = df_comparativo_balanceamento[col_bin]
        valores_esp = df_comparativo_balanceamento[col_esp]

        teste = calcular_teste_pareado(valores_bin, valores_esp)

        media_bin = valores_bin.mean()
        media_esp = valores_esp.mean()

        variacao_percentual = (
            (media_esp - media_bin) / media_bin * 100
            if media_bin != 0
            else np.nan
        )

        desvio_diferenca = teste["diferenca_desvio_padrao"]

        cohen_d = (
            teste["diferenca_media"] / desvio_diferenca
            if desvio_diferenca not in (0, None) and not pd.isna(desvio_diferenca)
            else np.nan
        )

        linhas_resumo_comparativo.append({
            "metrica": metrica,
            "media_bin_packing": media_bin,
            "desvio_padrao_bin_packing": valores_bin.std(ddof=1),
            "ic95_bin_packing": intervalo_confianca_95(valores_bin),
            "media_espacial": media_esp,
            "desvio_padrao_espacial": valores_esp.std(ddof=1),
            "ic95_espacial": intervalo_confianca_95(valores_esp),
            "variacao_percentual_espacial_vs_bin_packing": variacao_percentual,
            "n_pares": teste["n_pares"],
            "diferenca_media": teste["diferenca_media"],
            "estatistica_t": teste["estatistica_t"],
            "p_valor_t": teste["p_valor_t"],
            "estatistica_wilcoxon": teste["estatistica_wilcoxon"],
            "p_valor_wilcoxon": teste["p_valor_wilcoxon"],
            "d_cohen_pareado": cohen_d
        })

    df_resumo_comparativo_balanceamento = pd.DataFrame(
        linhas_resumo_comparativo
    )

    df_resumo_comparativo_balanceamento.to_excel(
        caminho_tabela("comparativo_balanceamento_resumo.xlsx"),
        index=False
    )

    _variacao_distancia = df_resumo_comparativo_balanceamento.loc[
        df_resumo_comparativo_balanceamento["metrica"] == "distancia_km",
        "variacao_percentual_espacial_vs_bin_packing"
    ]

    if len(_variacao_distancia) > 0:
        log_etapa(
            f"Comparativo de balanceamento: distancia espacial vs. "
            f"bin-packing = {_variacao_distancia.iloc[0]:.1f}% de variacao "
            f"media"
        )

else:

    log_etapa(
        "Comparativo de balanceamento: nenhum dia processado - "
        "comparativo_balanceamento_resumo.xlsx nao gerado"
    )

df_ablation.to_excel(
    caminho_tabela("ablation_study_diario.xlsx"),
    index=False
)

resumo_baselines.to_excel(
    caminho_tabela("resumo_baselines.xlsx"),
    index=False
)

df_resumo_baselines.to_excel(
    caminho_tabela("resumo_baselines_ic95.xlsx"),
    index=False
)

df_resumo_ablation.to_excel(
    caminho_tabela("resumo_ablation_ic95.xlsx"),
    index=False
)

df_ganhos_ablation.to_excel(
    caminho_tabela("ganhos_incrementais_ablation.xlsx"),
    index=False
)

df_resumo_ganhos_ablation.to_excel(
    caminho_tabela("resumo_ganhos_incrementais_ic95.xlsx"),
    index=False
)

df_2opt.to_excel(
    caminho_tabela("comparativo_antes_depois_2opt.xlsx"),
    index=False
)

df_cargas_boxplot.to_excel(
    caminho_tabela("cargas_equipes_boxplot.xlsx"),
    index=False
)

comparativo_ganhos.to_excel(
    caminho_tabela("ganhos_percentuais.xlsx"),
    index=False
)

resumo_balanceamento.to_excel(
    caminho_tabela("resumo_balanceamento.xlsx"),
    index=False
)

teste_wilcoxon.to_excel(
    caminho_tabela("testes_estatisticos.xlsx"),
    index=False
)

teste_significancia_estatistica.to_excel(
    caminho_tabela("testes_significancia_estatistica.xlsx"),
    index=False
)

ablation_study.to_excel(
    caminho_tabela("ablation_study.xlsx"),
    index=False
)

ganho_balanceamento.to_excel(
    caminho_tabela("ganho_balanceamento.xlsx"),
    index=False
)

ganho_aco.to_excel(
    caminho_tabela("ganho_aco.xlsx"),
    index=False
)

ganho_2opt_experimento.to_excel(
    caminho_tabela("ganho_2opt.xlsx"),
    index=False
)

tabela_artigo.to_excel(
    caminho_tabela("tabela_artigo.xlsx"),
    index=False
)

# ==========================================================
# DISPERSAO DAS REPLICACOES DO ACO NO PIPELINE PRINCIPAL
# ==========================================================
# Ver N_REPLICACOES_PIPELINE e `avaliar_baselines_rotas`: a replicacao
# de indice 0 e o resultado OFICIAL (ja usado em todo o restante do
# pipeline, sem alteracao); as demais servem so para medir a dispersao
# do ACO entre execucoes, isolando-a da variabilidade entre dias.

df_replicacoes_aco_pipeline = pd.DataFrame(
    REGISTROS_REPLICACOES_ACO_PIPELINE
)

df_replicacoes_aco_pipeline.to_excel(
    caminho_tabela("replicacoes_aco_pipeline.xlsx"),
    index=False
)

if len(df_replicacoes_aco_pipeline) > 0:

    distancia_total_por_replicacao = (
        df_replicacoes_aco_pipeline
        .groupby(["dia", "replicacao"])["distancia_aco_2opt_km"]
        .sum()
        .reset_index()
    )

    registros_resumo_replicacoes = []

    for dia_resumo, grupo in distancia_total_por_replicacao.groupby("dia"):

        media = grupo["distancia_aco_2opt_km"].mean()

        desvio_padrao = grupo["distancia_aco_2opt_km"].std(ddof=1)

        coeficiente_variacao = (
            desvio_padrao / media
            if media > 0 and pd.notna(desvio_padrao)
            else 0
        )

        registros_resumo_replicacoes.append({
            "dia": dia_resumo,
            "n_replicacoes": grupo["replicacao"].nunique(),
            "distancia_total_media_km": media,
            "distancia_total_desvio_padrao_km": desvio_padrao,
            "coeficiente_variacao_distancia_total": coeficiente_variacao
        })

    df_resumo_replicacoes_aco = pd.DataFrame(registros_resumo_replicacoes)

else:

    df_resumo_replicacoes_aco = pd.DataFrame()

df_resumo_replicacoes_aco.to_excel(
    caminho_tabela("replicacoes_aco_resumo.xlsx"),
    index=False
)

# ==========================================================
# CARACTERIZACAO DA BASE (Tabela do TCC)
# ==========================================================
# Funil de limpeza (uma linha por etapa de filtragem, ver
# `registrar_etapa_funil`) + caracterizacao da base final, usada e
# efetivamente processada pelo pipeline.

df_funil_limpeza = pd.DataFrame(registros_funil_limpeza)

pontos_por_dia = base.groupby("dia").size()

equipes_por_dia = df_balanceamento[
    df_balanceamento["cenario"] == "Com balanceamento"
]["equipes_usadas"]

df_base_final = pd.DataFrame([
    {"indicador": "total_ocorrencias", "valor": len(base)},
    {"indicador": "numero_dias", "valor": base["dia"].nunique()},
    {"indicador": "numero_zips_distintos", "valor": base["zip"].nunique()},
    {"indicador": "pontos_por_dia_minimo", "valor": pontos_por_dia.min()},
    {"indicador": "pontos_por_dia_medio", "valor": pontos_por_dia.mean()},
    {"indicador": "pontos_por_dia_maximo", "valor": pontos_por_dia.max()},
    {"indicador": "equipes_por_dia_minimo", "valor": equipes_por_dia.min()},
    {"indicador": "equipes_por_dia_medio", "valor": equipes_por_dia.mean()},
    {"indicador": "equipes_por_dia_maximo", "valor": equipes_por_dia.max()},
    {"indicador": "periodo_data_inicial", "valor": min(dias)},
    {"indicador": "periodo_data_final", "valor": max(dias)}
])

with pd.ExcelWriter(caminho_tabela("caracterizacao_base.xlsx")) as escritor:

    df_funil_limpeza.to_excel(
        escritor,
        sheet_name="Funil de limpeza",
        index=False
    )

    df_base_final.to_excel(
        escritor,
        sheet_name="Base final",
        index=False
    )

tempo_extra_replicacoes_segundos = (
    df_replicacoes_aco_pipeline[
        df_replicacoes_aco_pipeline["replicacao"] > 0
    ]["tempo_execucao_segundos"].sum()
    if len(df_replicacoes_aco_pipeline) > 0
    else 0
)

log_etapa(
    f"Replicacoes do ACO no pipeline (N_REPLICACOES_PIPELINE="
    f"{N_REPLICACOES_PIPELINE}): tempo adicional total "
    f"{tempo_extra_replicacoes_segundos / 60:.1f} min alem da replicacao "
    f"oficial (indice 0)"
)

pd.DataFrame([
    {
        "termo": "distancia_total_km",
        "peso": PESO_DISTANCIA,
        "penalidade": "distancia percorrida"
    },
    {
        "termo": "jornada_excedida_horas",
        "peso": PESO_JORNADA_EXCEDIDA,
        "penalidade": "horas acima da jornada maxima"
    },
    {
        "termo": "tempo_ocioso_total_horas",
        "peso": PESO_OCIOSIDADE,
        "penalidade": "ociosidade agregada das equipes"
    },
    {
        "termo": "diferenca_mais_menos_carregada_horas",
        "peso": PESO_DESBALANCEAMENTO,
        "penalidade": "desbalanceamento entre equipes"
    }
]).to_excel(
    caminho_tabela("funcao_objetivo_formalizada.xlsx"),
    index=False
)

if plt is not None:

    if (
        len(df_resumo_baselines) > 0
        and
        "metodo" in df_resumo_baselines.columns
    ):

        os.makedirs("graficos", exist_ok=True)

        ordem_metodos_baselines = [
            "FIFO",
            "Nearest Neighbor",
            "ACO sem 2OPT",
            "ACO + 2OPT"
        ]

        df_resumo_baselines_plot = df_resumo_baselines[
            df_resumo_baselines["metodo"].isin(ordem_metodos_baselines)
        ].copy()

        df_resumo_baselines_plot["metodo"] = pd.Categorical(
            df_resumo_baselines_plot["metodo"],
            categories=ordem_metodos_baselines,
            ordered=True
        )

        df_resumo_baselines_plot = df_resumo_baselines_plot.sort_values(
            "metodo"
        )

        plt.figure(figsize=(8, 4.5))
        plt.bar(
            df_resumo_baselines_plot["metodo"].astype(str),
            df_resumo_baselines_plot["distancia_total_km_media"],
            yerr=df_resumo_baselines_plot["distancia_total_km_ic95"],
            capsize=5,
            color="#457b9d"
        )
        plt.ylabel("Distancia total media (km)")
        plt.title("Comparativo de baselines de rota - IC 95%")
        plt.tight_layout()
        plt.savefig(
            caminho_grafico("comparativo_baselines_ic95.png"),
            dpi=160
        )
        plt.close()

    if len(df_2opt) > 0:

        df_2opt_plot = df_2opt[
            ["antes_2opt_km", "depois_2opt_km"]
        ].mean()

        plt.figure(figsize=(7, 4))
        df_2opt_plot.plot(kind="bar", color=["#8a8f98", "#2a9d8f"])
        plt.ylabel("Distancia media por equipe (km)")
        plt.title("Antes e depois do 2OPT")
        plt.tight_layout()
        plt.savefig(
            caminho_grafico("grafico_antes_depois_2opt.png"),
            dpi=160
        )
        plt.close()

    if len(df_cargas_boxplot) > 0:

        # Figura de distribuicao da carga por equipe (Figura 2 do TCC).
        # v11: sem winsorizacao; outliers visiveis; ordem Sem -> Com (a
        # mesma da tabela de balanceamento); o losango e a MEDIA DAS
        # MEDIAS DIARIAS - exatamente o valor de carga_media_horas_media
        # em resumo_balanceamento.xlsx -, e nao a media agrupada de todas
        # as equipes (que pondera mais os dias com mais equipes e, na v10,
        # gerava 7,57 h na figura contra 7,56 h na tabela). O painel da
        # direita amplia a faixa de 7 a 8 horas, onde se concentra a
        # maior parte das equipes.
        from matplotlib.ticker import FuncFormatter

        formatador_virgula = FuncFormatter(
            lambda valor, _pos: f"{valor:.1f}".replace(".", ",")
        )

        ordem_cenarios = [
            cenario
            for cenario in ["Sem balanceamento", "Com balanceamento"]
            if cenario in df_cargas_boxplot["cenario"].unique()
        ]

        dados_boxplot = [
            df_cargas_boxplot.loc[
                df_cargas_boxplot["cenario"] == cenario,
                "carga_horas"
            ].values
            for cenario in ordem_cenarios
        ]

        medias_diarias = [
            df_cargas_boxplot.loc[
                df_cargas_boxplot["cenario"] == cenario
            ].groupby("dia")["carga_horas"].mean().mean()
            for cenario in ordem_cenarios
        ]

        cores_cenarios = {
            "Sem balanceamento": "#9aa0a8",
            "Com balanceamento": "#1f7f7a"
        }

        figura, eixos = plt.subplots(
            1, 2,
            figsize=(11, 5),
            gridspec_kw={"width_ratios": [1, 1]}
        )

        for eixo, limites, titulo in (
            (eixos[0], None, "Distribuição completa"),
            (eixos[1], (7.0, 8.05), "Ampliação: 7 a 8 horas")
        ):
            caixas = eixo.boxplot(
                dados_boxplot,
                showfliers=True,
                patch_artist=True,
                widths=0.5,
                flierprops={
                    "marker": "o",
                    "markersize": 2.5,
                    "alpha": 0.35,
                    "markerfacecolor": "#555555",
                    "markeredgecolor": "none"
                },
                medianprops={"color": "#222222", "linewidth": 1.8}
            )
            for caixa, cenario in zip(caixas["boxes"], ordem_cenarios):
                caixa.set_facecolor(cores_cenarios[cenario])
                caixa.set_alpha(0.85)
            eixo.scatter(
                range(1, len(ordem_cenarios) + 1),
                medias_diarias,
                marker="D",
                s=60,
                color="#e07b39",
                zorder=4,
                label="Média (média das médias diárias)"
            )
            for posicao, media in enumerate(medias_diarias, start=1):
                if limites is None or limites[0] <= media <= limites[1]:
                    eixo.annotate(
                        f"{media:.2f} h".replace(".", ","),
                        (posicao, media),
                        xytext=(12, -4),
                        textcoords="offset points",
                        color="#e07b39",
                        fontweight="bold"
                    )
            eixo.set_xticks(range(1, len(ordem_cenarios) + 1))
            eixo.set_xticklabels(ordem_cenarios)
            eixo.yaxis.set_major_formatter(formatador_virgula)
            eixo.set_ylabel("Carga por equipe (horas)")
            eixo.set_title(titulo)
            eixo.spines["top"].set_visible(False)
            eixo.spines["right"].set_visible(False)
            if limites is not None:
                eixo.set_ylim(*limites)

        figura.legend(
            *eixos[0].get_legend_handles_labels(),
            loc="lower center",
            ncol=1,
            frameon=False,
            fontsize=9
        )

        figura.suptitle(
            "Carga por equipe, sem e com balanceamento",
            fontweight="bold"
        )
        figura.tight_layout(rect=(0, 0.06, 1, 1))
        figura.savefig(
            caminho_grafico("boxplot_carga_por_equipe.png"),
            dpi=200
        )
        plt.close(figura)

        # Distribuicao da carga por faixa de horas (Figura 3 do TCC),
        # gerada pelo proprio script (na v10 a figura do documento foi
        # feita a parte). Faixas em horas e minutos, como no texto.
        faixas_carga = [0, 2, 4, 6, 7, 7.5, 7.75, 7.9, 7.95, 8.0001]

        labels_faixas = [
            "0h00-2h00",
            "2h00-4h00",
            "4h00-6h00",
            "6h00-7h00",
            "7h00-7h30",
            "7h30-7h45",
            "7h45-7h54",
            "7h54-7h57",
            "7h57-8h00"
        ]

        df_histograma_carga = df_cargas_boxplot.copy()

        df_histograma_carga["faixa_carga"] = pd.cut(
            df_histograma_carga["carga_horas"],
            bins=faixas_carga,
            labels=labels_faixas,
            include_lowest=True,
            right=False
        )

        tabela_histograma_carga = pd.crosstab(
            df_histograma_carga["faixa_carga"],
            df_histograma_carga["cenario"]
        ).reindex(labels_faixas).fillna(0)

        tabela_histograma_carga = tabela_histograma_carga[
            [
                cenario
                for cenario in ["Sem balanceamento", "Com balanceamento"]
                if cenario in tabela_histograma_carga.columns
            ]
        ]

        # Exporta tambem a tabela da figura (contagem e percentual por
        # cenario), para que os percentuais citados no texto sejam
        # rastreaveis.
        tabela_histograma_percentual = (
            tabela_histograma_carga
            / tabela_histograma_carga.sum()
            * 100
        ).add_suffix(" (%)")

        pd.concat(
            [tabela_histograma_carga, tabela_histograma_percentual],
            axis=1
        ).reset_index().to_excel(
            caminho_tabela("distribuicao_carga_por_faixa.xlsx"),
            index=False
        )

        ax = tabela_histograma_carga.plot(
            kind="bar",
            figsize=(10, 5),
            color=[
                {"Sem balanceamento": "#9aa0a8", "Com balanceamento": "#1f7f7a"}[cenario]
                for cenario in tabela_histograma_carga.columns
            ],
            width=0.8
        )

        ax.set_title(
            "Distribuição da carga por equipe por faixa de horas, "
            "sem e com balanceamento",
            fontweight="bold"
        )
        ax.set_xlabel("Faixa de carga diária")
        ax.set_ylabel("Quantidade de equipes")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title=None, frameon=False)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            caminho_grafico("histograma_carga_por_equipe.png"),
            dpi=200
        )
        plt.close()

    if len(df_resumo_ablation) > 0:

        df_plot = df_resumo_ablation.sort_values(
            "funcao_objetivo_media"
        )

        plt.figure(figsize=(10, 4))
        plt.bar(
            df_plot["cenario"],
            df_plot["funcao_objetivo_media"],
            color="#457b9d"
        )
        plt.ylabel("Funcao objetivo media")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            caminho_grafico("grafico_ablation_funcao_objetivo.png"),
            dpi=160
        )
        plt.close()
else:

    log_etapa(
        "matplotlib nao encontrado; graficos PNG nao foram gerados"
    )

log_etapa(
    f"Excel exportado em {time.perf_counter() - inicio:.1f}s"
)

print("✔ EXPORTAÇÕES FINALIZADAS")

# ==========================================================
# HTML INTERATIVO
# ==========================================================
# Gera um único arquivo HTML autocontido (usando a biblioteca
# JavaScript Leaflet, carregada via CDN) que permite explorar as rotas
# otimizadas (ACO + 2-opt) de qualquer um dos dias simulados,
# sobrepostas ao mapa real de Nova York. O usuário escolhe um dia e
# quais equipes exibir; cada equipe é desenhada com uma cor diferente,
# usando as coordenadas de rua reais (obtidas via OSMnx/`ox.shortest_path`
# durante o loop principal), não a linha reta da distância planar -
# essa visualização é só para fins de apresentação/conferência visual,
# não afeta os cálculos de distância/tempo usados nas métricas.

print("\nCRIANDO HTML...")

inicio = time.perf_counter()

centro_lat = base["lat"].mean()
centro_lon = base["lon"].mean()

html = f"""
<!DOCTYPE html>
<html>

<head>

<meta charset='utf-8'/>

<title>Rotas Operacionais ACO</title>

<link rel='stylesheet'
href='https://unpkg.com/leaflet/dist/leaflet.css'/>

<script
src='https://unpkg.com/leaflet/dist/leaflet.js'>
</script>

<style>

body {{
    margin: 0;
}}

#map {{
    width: 100%;
    height: 100vh;
}}

.controls {{

    position: absolute;

    top: 10px;
    left: 10px;

    z-index: 9999;

    background: white;

    padding: 12px;

    border-radius: 10px;

    box-shadow: 0 0 15px rgba(0,0,0,0.3);

    font-family: Arial;
}}

.legend {{
    margin-top: 10px;
}}

#equipes {{
    width: 150px;
}}

.team-actions {{
    margin-top: 6px;
}}

.legend-item {{
    margin-bottom: 5px;
}}

.legend-color {{

    display: inline-block;

    width: 12px;
    height: 12px;

    margin-right: 5px;
}}

</style>

</head>

<body>

<div class='controls'>

<h3>Rotas Operacionais</h3>

<label>Dia:</label>

<select id='dia' onchange='atualizarEquipes()'></select>

<br><br>

<label>Equipes:</label>

<br>

<select id='equipes' multiple size='8'></select>

<div class='team-actions'>

<button onclick='selecionarTodasEquipes()'>
Todas
</button>

<button onclick='limparEquipes()'>
Limpar
</button>

</div>

<br><br>

<button onclick='plotar()'>
Plotar Rotas
</button>

<div class='legend' id='legend'></div>

</div>

<div id='map'></div>

<script>

var rotas = {json.dumps(rotas_json)};

var cores = [

    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "black",
    "brown",
    "pink",
    "darkred",
    "cadetblue",
    "darkgreen",
    "gray",
    "gold"
];

var map = L.map('map').setView(
    [{centro_lat}, {centro_lon}],
    10
);

L.tileLayer(
    'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{
        maxZoom: 19
    }}
).addTo(map);

var dias = [...new Set(
    rotas.map(r => r.dia)
)];

dias.forEach(d => {{

    var option =
    document.createElement('option');

    option.value = d;
    option.text = d;

    document
    .getElementById('dia')
    .appendChild(option);
}});

function atualizarEquipes() {{

    var dia = document
        .getElementById('dia').value;

    var selectEquipes = document
        .getElementById('equipes');

    selectEquipes.innerHTML = "";

    var equipesDia = [...new Set(
        rotas
            .filter(r => r.dia == dia)
            .map(r => r.equipe)
    )];

    equipesDia.forEach(equipe => {{

        var option =
        document.createElement('option');

        option.value = equipe;
        option.text = "Equipe " + equipe;
        option.selected = true;

        selectEquipes.appendChild(option);
    }});
}}

atualizarEquipes();

function selecionarTodasEquipes() {{

    Array.from(
        document.getElementById('equipes').options
    ).forEach(option => {{
        option.selected = true;
    }});
}}

function limparEquipes() {{

    Array.from(
        document.getElementById('equipes').options
    ).forEach(option => {{
        option.selected = false;
    }});
}}

var layersRotas = [];

var layersMarkers = [];

function limparMapa() {{

    layersRotas.forEach(l => {{
        map.removeLayer(l);
    }});

    layersMarkers.forEach(l => {{
        map.removeLayer(l);
    }});

    layersRotas = [];
    layersMarkers = [];
}}

function plotar() {{

    limparMapa();

    document.getElementById(
        "legend"
    ).innerHTML = "";

    var dia = document
        .getElementById('dia').value;

    var equipesSelecionadas = Array.from(
        document.getElementById('equipes').selectedOptions
    ).map(option => option.value);

    var rotasDia = rotas.filter(
        r => (
            r.dia == dia
            &&
            equipesSelecionadas.includes(r.equipe)
        )
    );

    if (rotasDia.length == 0) {{

        alert("Nenhuma rota encontrada para a seleção");

        return;
    }}

    rotasDia.forEach((rota, idx) => {{

        var cor = cores[
            idx % cores.length
        ];

        var poly = L.polyline(

            rota.coords,

            {{
                color: cor,
                weight: 5
            }}

        ).addTo(map);

        layersRotas.push(poly);

        rota.pontos.forEach(p => {{

            var marker = L.circleMarker(

                [p.lat, p.lon],

                {{
                    radius: 6,
                    color: cor,
                    fillColor: cor,
                    fillOpacity: 1
                }}

            ).addTo(map);

            marker.bindPopup(

                "<b>Equipe:</b> " +

                rota.equipe +

                "<br>" +

                "<b>Ordem:</b> " +

                p.ordem +

                "<br>" +

                "<b>ZIP:</b> " +

                p.zip
            );

            layersMarkers.push(marker);
        }});

        var legenda = document.createElement(
            "div"
        );

        legenda.className = "legend-item";

        legenda.innerHTML =

            "<span class='legend-color' " +

            "style='background:" + cor + "'></span>" +

            "Equipe " + rota.equipe;

        document.getElementById(
            "legend"
        ).appendChild(legenda);

    }});

    var grupo = L.featureGroup(
        layersRotas
    );

    map.fitBounds(
        grupo.getBounds()
    );
}}

</script>

</body>
</html>
"""

with open(

    "rotas_interativas.html",

    "w",

    encoding="utf-8"

) as f:

    f.write(html)

print("✔ rotas_interativas.html")

log_etapa(
    f"HTML criado em {time.perf_counter() - inicio:.1f}s"
)

print("\nPROCESSAMENTO FINALIZADO")

# ==========================================================
# ANALISE DE SENSIBILIDADE DO ACO (aditiva, ao final do pipeline)
# ==========================================================

if EXECUTAR_SENSIBILIDADE_ACO:

    analisar_sensibilidade_aco()

if EXECUTAR_SENSIBILIDADE_TOLERANCIA:

    analisar_sensibilidade_tolerancia()

# ==========================================================
# MANIFESTO DE SAIDAS
# ==========================================================
# Lista, para auditoria, TODOS os arquivos gerados dentro de PASTA_SAIDA
# (tamanho em bytes e data/hora de geracao), o tempo total de execucao do
# script e os valores das principais constantes usadas na execucao - ver
# secao 4 das instrucoes de alteracao do script.

_arquivos_manifesto = []

for _raiz, _dirs, _arquivos in os.walk(PASTA_SAIDA):

    for _nome_arquivo in sorted(_arquivos):

        if _nome_arquivo == "MANIFESTO.txt":
            continue

        _caminho_completo = os.path.join(_raiz, _nome_arquivo)

        _stat = os.stat(_caminho_completo)

        _arquivos_manifesto.append({
            "caminho": os.path.relpath(_caminho_completo, PASTA_SAIDA),
            "tamanho_bytes": _stat.st_size,
            "gerado_em": pd.Timestamp.fromtimestamp(
                _stat.st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")
        })

_tempo_total_execucao_segundos = time.perf_counter() - INICIO_SCRIPT

with open(
    os.path.join(PASTA_SAIDA, "MANIFESTO.txt"),
    "w",
    encoding="utf-8"
) as _manifesto:

    _manifesto.write("MANIFESTO DE SAIDAS - PIPELINE ACO+2OPT (TCC) - v11\n")
    _manifesto.write("=" * 60 + "\n\n")

    _manifesto.write(
        f"Tempo total de execucao: "
        f"{_tempo_total_execucao_segundos:.1f}s "
        f"({_tempo_total_execucao_segundos / 3600:.2f}h)\n\n"
    )

    _manifesto.write("Principais constantes usadas nesta execucao:\n")

    for _nome_constante in [
        "SEMENTE_ALEATORIA",
        "JORNADA_MAXIMA_HORAS",
        "JORNADA_ALVO_HORAS",
        "JORNADA_MINIMA_DESEJADA_HORAS",
        "VELOCIDADE_MEDIA",
        "TEMPO_BUFFER_MINUTOS",
        "ACO_FORMIGAS",
        "ACO_ITERACOES",
        "ACO_ALPHA",
        "ACO_BETA",
        "ACO_EVAPORACAO",
        "ACO_Q",
        "PESO_DISTANCIA",
        "PESO_JORNADA_EXCEDIDA",
        "PESO_OCIOSIDADE",
        "PESO_DESBALANCEAMENTO",
        "ORTOOLS_LIMITE_SEGUNDOS",
        "N_REPLICACOES_PIPELINE",
        "REPROCESSAR_TUDO",
        "LIMITE_DIAS_TESTE",
        "DIAS_ESPECIFICOS_TESTE",
        "KM_POR_GRAU_LATITUDE",
        "LATITUDE_REFERENCIA_GRAUS",
        "KM_POR_GRAU_LONGITUDE",
        "TOLERANCIA_KM_POR_HORA_OCIOSA",
        "SA_TEMPERATURA_INICIAL",
        "SA_RESFRIAMENTO",
        "SA_ITERACOES",
        "N_DIAS_SENSIBILIDADE",
        "N_REPLICACOES",
        "GRADE_TOLERANCIA_KM_POR_HORA_OCIOSA",
        "ORTOOLS_DISPONIVEL"
    ]:
        _manifesto.write(
            f"  {_nome_constante} = {globals().get(_nome_constante)}\n"
        )

    _manifesto.write(
        f"\nArquivos gerados em '{PASTA_SAIDA}/' "
        f"({len(_arquivos_manifesto)} arquivos):\n\n"
    )

    for _info in _arquivos_manifesto:
        _manifesto.write(
            f"  {_info['caminho']:<70} "
            f"{_info['tamanho_bytes']:>12} bytes  "
            f"{_info['gerado_em']}\n"
        )

log_etapa(
    f"MANIFESTO.txt gerado com {len(_arquivos_manifesto)} arquivos - "
    f"tempo total de execucao: {_tempo_total_execucao_segundos:.1f}s"
)
