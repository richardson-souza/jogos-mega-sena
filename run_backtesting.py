import pandas as pd
import random
import argparse
import numpy as np
from datetime import timedelta
from src.genetic_algorithm import load_historical_games, run_evolution, generate_random_portfolio
from src.association_rules import mine_frequent_itemsets, train_kmeans_clusters

def evaluate_games(games: list, draws: list) -> dict:
    results = {3: 0, 4: 0, 5: 0, 6: 0}
    for draw in draws:
        draw_set = set(draw)
        for game in games:
            hits = len(draw_set.intersection(set(game)))
            if hits >= 3:
                results[hits] += 1
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monte-carlo', action='store_true', help='Executar teste de estresse Monte Carlo (1M sorteios)')
    args = parser.parse_args()

    filepath = "data/processed/mega_sena_features.csv"
    print("Iniciando Módulo de Backtesting...")
    
    df = pd.read_csv(filepath)
    df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], format='%d/%m/%Y')
    
    # Split de teste (Último ano para out-of-sample)
    max_date = df['Data do Sorteio'].max()
    split_date = max_date - timedelta(days=365)
    
    df_train = df[df['Data do Sorteio'] <= split_date]
    df_test = df[df['Data do Sorteio'] > split_date]
    
    print(f"Período de Treino: {df_train['Data do Sorteio'].min().date()} até {split_date.date()} ({len(df_train)} sorteios)")
    print(f"Período de Teste (Out-of-Sample): {df_test['Data do Sorteio'].min().date()} até {max_date.date()} ({len(df_test)} sorteios)")
    
    # Criar set histórico de treino e frequencias para o GA
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_train = set()
    
    all_numbers_train = df_train[cols_bolas].values.flatten()
    all_numbers_train = all_numbers_train[~pd.isna(all_numbers_train)].astype(int)
    freq_train = (pd.Series(all_numbers_train).value_counts() / len(df_train)).to_dict()
    
    for _, row in df_train[cols_bolas].iterrows():
        historical_train.add(frozenset(row.dropna().astype(int)))
        
    out_of_sample_draws = df_test[cols_bolas].values.tolist()
    out_of_sample_draws = [[int(x) for x in draw if not pd.isna(x)] for draw in out_of_sample_draws]
        
    # --- 1. Extração de Conhecimento (Data Mining) ---
    print("\nExtraindo Conhecimento de Data Mining (Apriori/KMeans)...")
    itemsets_df = mine_frequent_itemsets(df_train, min_support=0.01)
    if not itemsets_df.empty:
        itemsets_df['length'] = itemsets_df['itemsets'].apply(len)
        valid_itemsets = [list(x) for x in itemsets_df[itemsets_df['length'] >= 2]['itemsets'].tolist()]
    else:
        valid_itemsets = []
        
    _, centroids = train_kmeans_clusters(df_train, n_clusters=5)
    
    # --- 2. Treinamento dos Modelos Ensemble ---
    print("Executando GA [Padrão]...")
    ga_standard = run_evolution(historical_train, freq_train, pop_size=50, generations=100, mutation_rate=0.05, method='standard')
    
    print("Executando GA [Boosted Initialization]...")
    ga_boosted = run_evolution(historical_train, freq_train, pop_size=50, generations=100, mutation_rate=0.05, method='boosted', itemsets=valid_itemsets)
    
    print("Executando GA [Stacking / Meta-Fitness]...")
    ga_stacked = run_evolution(historical_train, freq_train, pop_size=50, generations=100, mutation_rate=0.05, method='stacking', itemsets=valid_itemsets, centroids=centroids)
    
    print("Gerando jogos Aleatórios (Controle)...")
    random_games = generate_random_portfolio(10)
    
    # --- 3. Avaliação Out-of-Sample ---
    print("\nSimulando apostas no período de Teste (Último Ano)...")
    eval_std = evaluate_games(ga_standard, out_of_sample_draws)
    eval_boo = evaluate_games(ga_boosted, out_of_sample_draws)
    eval_stk = evaluate_games(ga_stacked, out_of_sample_draws)
    eval_rnd = evaluate_games(random_games, out_of_sample_draws)
    
    print("\n================ RESULTADOS DO BACKTESTING ================")
    print(f"{'Métrica':<15} | {'GA Padrão':<15} | {'GA Boosted':<15} | {'GA Stacked':<15} | {'Aposta Cega':<15}")
    print("-" * 83)
    for hits, label in [(3, 'Terno (Prox.)'), (4, 'Quadra'), (5, 'Quina'), (6, 'Sena')]:
        print(f"{label:<15} | {eval_std[hits]:<15} | {eval_boo[hits]:<15} | {eval_stk[hits]:<15} | {eval_rnd[hits]:<15}")
    print("===========================================================\n")

    if args.monte_carlo:
        print("\n================ ESTRESSE DE MONTE CARLO (1.000.000 Sorteios) ================")
        print("Gerando 1.000.000 de sorteios aleatórios sintéticos...")
        mc_draws = np.argsort(np.random.rand(1000000, 60), axis=1)[:, :6] + 1
        
        print("Avaliando bilhetes contra a base sintética vetorizada...")
        def evaluate_monte_carlo(games_list, draws_matrix):
            results = {3: 0, 4: 0, 5: 0, 6: 0}
            for game in games_list:
                mask = np.isin(draws_matrix, game)
                hits_per_draw = mask.sum(axis=1)
                for k in results.keys():
                    results[k] += int(np.sum(hits_per_draw == k))
            return results
            
        mc_std = evaluate_monte_carlo(ga_standard, mc_draws)
        mc_boo = evaluate_monte_carlo(ga_boosted, mc_draws)
        mc_stk = evaluate_monte_carlo(ga_stacked, mc_draws)
        mc_rnd = evaluate_monte_carlo(random_games, mc_draws)
        
        print(f"\n{'Métrica':<15} | {'GA Padrão':<15} | {'GA Boosted':<15} | {'GA Stacked':<15} | {'Aposta Cega':<15}")
        print("-" * 83)
        for hits in [3, 4, 5, 6]:
            label = f"{hits} Acertos"
            print(f"{label:<15} | {mc_std[hits]:<15} | {mc_boo[hits]:<15} | {mc_stk[hits]:<15} | {mc_rnd[hits]:<15}")
        
        print("==============================================================================\n")

if __name__ == "__main__":
    main()
