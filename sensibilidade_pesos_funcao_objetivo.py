"""Sensibilidade da ordenacao dos cenarios aos pesos da funcao objetivo (Tabela 7).
Recalcula F = wD*D + wJ*J + wO*O + wB*B para cada cenario, a partir de
ablation_study_diario.xlsx (v11), e verifica se a ordenacao das medias dos
cinco cenarios permanece igual a obtida com os pesos originais (1; 3; 0,8; 1,5)."""
import itertools, numpy as np, pandas as pd, sys
ad = pd.read_excel(sys.argv[1])
comp = ad.groupby("cenario")[["distancia_total_km", "jornada_excedida_total_horas",
                              "tempo_ocioso_total_horas", "diferenca_mais_menos_carregada_horas"]].mean()
X = comp.values
orig = np.array([1.0, 3.0, 0.8, 1.5])
ordem = lambda w: tuple(comp.index[np.argsort(X @ w, kind="stable")])
ref = ordem(orig)
assert np.allclose(X @ orig, ad.groupby("cenario").funcao_objetivo.mean().loc[comp.index].values)
mult = [0.5, 1, 1.5, 2, 3, 6]
grade = [orig * np.array(m) for m in itertools.product(mult, repeat=4)]
rng = np.random.default_rng(42)
aleat = [rng.uniform(0, 20, 4) for _ in range(2000)]
extremos = {"Apenas distância (1,0; 0; 0; 0)": np.array([1, 0, 0, 0.]),
            "Ociosidade e desbalanceamento anulados (1,0; 3,0; 0; 0)": np.array([1, 3, 0, 0.]),
            "Desbalanceamento dominante (1,0; 3,0; 0,8; 10,0)": np.array([1, 3, .8, 10.])}
g = sum(ordem(w) == ref for w in grade); a = sum(ordem(w) == ref for w in aleat)
linhas = [("Grade sistemática (0,5× a 6× dos pesos originais)", len(grade), g),
          ("Sorteio aleatório uniforme U(0; 20)", len(aleat), a)]
linhas += [(k, 1, int(ordem(w) == ref)) for k, w in extremos.items()]
out = pd.DataFrame(linhas, columns=["configuracao", "avaliadas", "ordenacao_identica"])
print("ordem de referencia:", ref); print(out.to_string()); print(comp.to_string())
if len(sys.argv) > 2: out.to_excel(sys.argv[2], index=False)
