import pandas as pd
import numpy as np
from src.genetic_algorithm import load_historical_games, run_evolution
from src.association_rules import mine_frequent_itemsets, train_kmeans_clusters

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default="data/processed/mega_sena_features.csv", help='Caminho do dataset')
    args = parser.parse_args()
    filepath = args.filepath
    print("Iniciando Análise de Sensibilidade (Grid Search)...")
    
    historical_train, freq_train, stats = load_historical_games(filepath)
    df = pd.read_csv(filepath)
        
    print("\nExtraindo Conhecimento de Data Mining (Apriori/KMeans)...")
    itemsets_df = mine_frequent_itemsets(df, min_support=0.01)
    if not itemsets_df.empty:
        itemsets_df['length'] = itemsets_df['itemsets'].apply(len)
        valid_itemsets = [list(x) for x in itemsets_df[itemsets_df['length'] >= 2]['itemsets'].tolist()]
    else:
        valid_itemsets = []
        
    _, centroids = train_kmeans_clusters(df, n_clusters=5)
    
    # Grid de Perfis
    profiles = {
        "Padrão": {'w_freq': 10.0, 'w_apriori': 5.0, 'w_kmeans': 0.5, 'w_hamming': 5.0, 'w_consec': 5.0, 'w_std': 5.0, 'w_zone': 5.0},
        "A: Espalhar": {'w_freq': 5.0, 'w_apriori': 1.0, 'w_kmeans': 0.1, 'w_hamming': 20.0, 'w_consec': 1.0, 'w_std': 1.0, 'w_zone': 1.0},
        "B: Tendência": {'w_freq': 5.0, 'w_apriori': 20.0, 'w_kmeans': 2.0, 'w_hamming': 1.0, 'w_consec': 5.0, 'w_std': 5.0, 'w_zone': 5.0},
        "C: Frequência": {'w_freq': 30.0, 'w_apriori': 5.0, 'w_kmeans': 0.5, 'w_hamming': 5.0, 'w_consec': 1.0, 'w_std': 1.0, 'w_zone': 1.0},
        "D: Dinâmico": None, # Tratado de forma especial
        "E: Físico": {'w_freq': 5.0, 'w_apriori': 5.0, 'w_kmeans': 0.5, 'w_hamming': 5.0, 'w_consec': 30.0, 'w_std': 30.0, 'w_zone': 30.0}
    }
    
    results_portfolios = {}
    
    for profile_name, weights in profiles.items():
        print(f"Executando GA Stacked - Perfil [{profile_name}]...")
        
        is_dynamic = True if profile_name == "D: Dinâmico" else False
        
        portfolio = run_evolution(
            historical_train, 
            freq_train, 
            pop_size=50, 
            generations=100, 
            mutation_rate=0.05, 
            method='stacking', 
            itemsets=valid_itemsets, 
            centroids=centroids,
            weights_dict=weights,
            dynamic_weights=is_dynamic,
            stats=stats
        )
        results_portfolios[profile_name] = portfolio
        
    print("\n================ ESTRESSE DE MONTE CARLO (1.000.000 Sorteios) ================")
    print("Gerando 1.000.000 de sorteios aleatórios sintéticos...")
    mc_draws = np.argsort(np.random.rand(1000000, 60), axis=1)[:, :6] + 1
    
    print("Avaliando os perfis de peso contra a base sintética vetorizada...")
    def evaluate_monte_carlo(games_list, draws_matrix):
        results = {3: 0, 4: 0, 5: 0, 6: 0}
        for game in games_list:
            mask = np.isin(draws_matrix, game)
            hits_per_draw = mask.sum(axis=1)
            for k in results.keys():
                results[k] += int(np.sum(hits_per_draw == k))
        return results
        
    evaluations = {}
    for name, port in results_portfolios.items():
        evaluations[name] = evaluate_monte_carlo(port, mc_draws)
        
    print(f"\n{'Métrica':<15} | {'Padrão':<10} | {'A (Espalhar)':<12} | {'B (Tendência)':<13} | {'C (Frequência)':<14} | {'D (Dinâmico)':<12} | {'E (Físico)':<10}")
    print("-" * 105)
    for hits in [3, 4, 5, 6]:
        label = f"{hits} Acertos"
        print(f"{label:<15} | {evaluations['Padrão'][hits]:<10} | {evaluations['A: Espalhar'][hits]:<12} | {evaluations['B: Tendência'][hits]:<13} | {evaluations['C: Frequência'][hits]:<14} | {evaluations['D: Dinâmico'][hits]:<12} | {evaluations['E: Físico'][hits]:<10}")
    
    print("=========================================================================================================\n")

if __name__ == "__main__":
    main()
