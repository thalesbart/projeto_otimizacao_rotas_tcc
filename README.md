# Balanceamento espacial e roteamento metaheurístico de equipes técnicas

Código do Trabalho de Conclusão de Curso do MBA em Data Science e Analytics (USP/Esalq):
**"Balanceamento Espacial e Roteamento Metaheurístico de Equipes Técnicas em Atendimento Emergencial"**,
de Thales Eduardo Pereira Bartolomeu, com orientação de Everton Gomede.

O modelo distribui as ocorrências diárias do Corpo de Bombeiros de Nova York (FDNY, 2019) entre equipes
por balanceamento espacial (k-means capacitado pela jornada de 8 h) e define a rota de cada equipe por
Ant Colony Optimization (Ant System) com refinamento 2OPT. Os resultados são comparados com FIFO,
Nearest Neighbor, Simulated Annealing e OR-Tools, com estudo de ablação e testes pareados
(t e Wilcoxon, correção de Holm-Bonferroni).

## Arquivamento (DOI)

| Conteúdo | DOI |
|---|---|
| Código (este repositório) | [10.5281/zenodo.23069024](https://doi.org/10.5281/zenodo.23069024) |
| Saídas, análises complementares e base tratada | [10.5281/zenodo.23069797](https://doi.org/10.5281/zenodo.23069797) |

## Arquivos

| Arquivo | Função |
|---|---|
| `otimizacao_rotas_aco_2opt.py` | Fluxo principal: limpeza e georreferenciamento, alocação (com e sem balanceamento), roteamento pelos seis métodos, ablação, testes estatísticos, análises de sensibilidade e gráficos. Gera a pasta `saidas_v11/`. |
| `analise_complementar_v11.py` | Análises complementares que não exigem rodar o fluxo principal de novo: efeito do balanceamento sob ACO + 2OPT e curva de convergência do ACO. Gera `saidas_complementares_v11/`. |
| `requirements.txt` / `environment.yml` | Versões exatas das bibliotecas usadas. |

## Ambiente

Python 3.10.20. Com conda:

```
conda env create -f environment.yml
conda activate tcc_rl
```

Ou com pip (Python 3.10):

```
python -m pip install -r requirements.txt
```

## Dados de entrada

1. **Despachos do FDNY** – *Fire Incident Dispatch Data* (NYC Open Data):
   https://data.cityofnewyork.us/Public-Safety/Fire-Incident-Dispatch-Data/8m42-w767 (registros de 2019).
2. **Códigos postais dos EUA** – TIGER/Line Shapefiles 2023, ZCTA5 (U.S. Census Bureau):
   https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/ (arquivo `tl_2023_us_zcta520`).

A base já tratada (`base_tratada.xlsx`) está no registro de saídas no Zenodo.

## Como executar

1. No início de `otimizacao_rotas_aco_2opt.py`, ajuste `PATH_BASE` (base do FDNY), `PATH_SHAPE`
   (shapefile) e `PATH_BASE_TRATADA` (onde a base tratada será gravada).
2. Rode:
   ```
   python otimizacao_rotas_aco_2opt.py
   ```
   A execução completa (364 dias, cinco réplicas do ACO e análises de sensibilidade) levou cerca de
   61,5 horas em um notebook Intel Core i7-1255U com 16 GB de RAM e Windows 11 Pro. O script grava
   checkpoints por dia e retoma de onde parou se for interrompido. Para um teste rápido, use
   `LIMITE_DIAS_TESTE = 5` ou `DIAS_ESPECIFICOS_TESTE = ["2019-01-01"]`.
3. Para as análises complementares, ajuste `CAMINHO_BASE_TRATADA` e `PASTA_V11` no início de
   `analise_complementar_v11.py` e rode:
   ```
   python analise_complementar_v11.py
   ```
   No mesmo equipamento, levou cerca de 4,5 horas.

Os algoritmos estocásticos usam sementes determinísticas por dia e equipe, de modo que a execução é
reprodutível.

## Licença

Código sob licença MIT (arquivo `LICENSE`). Os dados de entrada são públicos e seguem os termos de uso
de suas fontes.
