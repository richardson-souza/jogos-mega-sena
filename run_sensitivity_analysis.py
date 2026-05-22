import pandas as pd
import numpy as np
from src.genetic_algorithm import load_historical_games, run_evolution
from src.association_rules import mine_frequent_itemsets, train_kmeans_clusters

def main():
    filepath = "data/processed/mega_sena_features.csv"
    print("Iniciando Análise de Sensibilidade (Grid Search)...")
    
    df = pd.read_csv(filepath)
    
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_train = set()
    
    all_numbers_train = df[cols_bolas].values.flatten()
    all_numbers_train = all_numbers_train[~pd.isna(all_numbers_train)].astype(int)
    freq_train = (pd.Series(all_numbers_train).value_counts() / len(df)).to_dict()
    
    for _, row in df[cols_bolas].iterrows():
        historical_train.add(frozenset(row.dropna().astype(int)))
        
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
        "Padrão": {'w_freq': 10.0, 'w_apriori': 5.0, 'w_kmeans': 0.5, 'w_hamming': 5.0},
        "A: Foco Espalhamento": {'w_freq': 5.0, 'w_apriori': 1.0, 'w_kmeans': 0.1, 'w_hamming': 20.0},
        "B: Foco Tendência": {'w_freq': 5.0, 'w_apriori': 20.0, 'w_kmeans': 2.0, 'w_hamming': 1.0},
        "C: Foco Frequência": {'w_freq': 30.0, 'w_apriori': 5.0, 'w_kmeans': 0.5, 'w_hamming': 5.0}
    }
    
    results_portfolios = {}
    
    for profile_name, weights in profiles.items():
        print(f"Executando GA Stacked - Perfil [{profile_name}]...")
        portfolio = run_evolution(
            historical_train, 
            freq_train, 
            pop_size=50, 
            generations=100, 
            mutation_rate=0.05, 
            method='stacking', 
            itemsets=valid_itemsets, 
            centroids=centroids,
            weights_dict=weights
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
        
    print(f"\n{'Métrica':<15} | {'Padrão':<15} | {'A (Espalhar)':<15} | {'B (Tendência)':<15} | {'C (Frequência)':<15}")
    print("-" * 85)
    for hits in [3, 4, 5, 6]:
        label = f"{hits} Acertos"
        print(f"{label:<15} | {evaluations['Padrão'][hits]:<15} | {evaluations['A: Foco Espalhamento'][hits]:<15} | {evaluations['B: Foco Tendência'][hits]:<15} | {evaluations['C: Foco Frequência'][hits]:<15}")
    
    print("====================================================================================\n")

if __name__ == "__main__":
    main()
