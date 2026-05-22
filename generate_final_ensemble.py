import pandas as pd
import numpy as np
from src.genetic_algorithm import load_historical_games, run_evolution
from src.association_rules import generate_apriori_kmeans_games

def main():
    filepath = "data/processed/mega_sena_features.csv"
    
    print("=========================================================")
    print(" 🚀 INICIANDO GERAÇÃO DO ENSEMBLE PARA O CONCURSO 3010 🚀 ")
    print("=========================================================\n")
    
    df_train = pd.read_csv(filepath)
    
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_train, freq_train, stats = load_historical_games(filepath)
    
    print(f"[Info] Estatísticas base: Média do Desvio Padrão = {stats['std_mean']:.2f}")
    
    print("[1/2] Acionando Algoritmo Genético de Portfólios (Otimização Combinatória + Frequência)...")
    ga_games = run_evolution(historical_train, freq_train, pop_size=50, generations=100, mutation_rate=0.05, method='stacking', dynamic_weights=True, stats=stats)
    
    print("[2/2] Acionando Motor B (Apriori + K-Means Latent Clustering)...")
    apriori_games = generate_apriori_kmeans_games(df_train, num_games=10, n_clusters=5)
    
    print("\n================ OS 20 BILHETES DE OURO ================")
    print("\n--- Estratégia A: Algoritmo Genético (Portfólio Otimizado) ---")
    for i, game in enumerate(ga_games, 1):
        formatted_game = " - ".join([f"{n:02d}" for n in sorted(game)])
        print(f"Bilhete A{i:02d}: [ {formatted_game} ]")
        
    print("\n--- Estratégia B: Associação (Apriori) + K-Means (Wheeling System) ---")
    for i, game in enumerate(apriori_games, 1):
        formatted_game = " - ".join([f"{n:02d}" for n in sorted(game)])
        print(f"Bilhete B{i:02d}: [ {formatted_game} ]")
        
    print("\n--- Validação Física do Portfólio GA (Estratégia A) ---")
    import numpy as np
    stds = [np.std(game, ddof=1) for game in ga_games]
    avg_std = np.mean(stds)
    print(f"Desvio Padrão Médio dos Bilhetes Gerados: {avg_std:.2f} (Alvo Base: {stats['std_mean']:.2f})")
    
    print("\n================ BOA SORTE NO SORTEIO ESPECIAL! ================\n")

if __name__ == "__main__":
    main()
