import pandas as pd
import random
import argparse
import numpy as np
from datetime import timedelta
from src.genetic_algorithm import load_historical_games, run_evolution, generate_random_portfolio
from src.association_rules import generate_apriori_kmeans_games

def evaluate_games(games: list, draws: list) -> dict:
    """
    Simula os jogos contra uma lista de sorteios reais.
    Retorna o total de Quadras (4), Quinas (5) e Senas (6), além de Ternos (3) para métrica de proximidade.
    """
    results = {3: 0, 4: 0, 5: 0, 6: 0}
    for draw in draws:
        draw_set = set(draw)
        for game in games:
            hits = len(set(game).intersection(draw_set))
            if hits in results:
                results[hits] += 1
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monte-carlo', action='store_true', help='Executar teste de estresse Monte Carlo (1M sorteios)')
    args = parser.parse_args()

    filepath = "data/processed/mega_sena_features.csv"
    print("Iniciando Módulo de Backtesting...")
    
    # 1. Carregamento e Time-Split
    df = pd.read_csv(filepath)
    df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], format='%d/%m/%Y')
    df = df.sort_values(by='Data do Sorteio')
    
    max_date = df['Data do Sorteio'].max()
    split_date = max_date - pd.DateOffset(years=1)
    
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
        
    # Extrair lista de sorteios reais do teste
    draws_test = []
    for _, row in df_test[cols_bolas].iterrows():
        draws_test.append(row.dropna().astype(int).tolist())
        
    # 2. Gerar Jogos Otimizados via GA
    print("\nExecutando Algoritmo Genético (apenas no conjunto de treino)...")
    # Reduzido pop_size e generations pois agora o indivíduo é um portfólio inteiro
    ga_games = run_evolution(historical_train, freq_train, pop_size=50, generations=100, mutation_rate=0.05)
    print("Concluído. 10 jogos gerados.")
    
    # 3. Gerar Jogos Aleatórios (Aposta Cega)
    print("Gerando jogos Aleatórios (Cegos)...")
    random_games = generate_random_portfolio(10)
    
    # 3.5 Gerar Jogos do Motor B (Apriori + KMeans)
    print("Executando Motor B (Apriori + K-Means)... isso pode demorar um pouco devido à mineração.")
    apriori_games = generate_apriori_kmeans_games(df_train, num_games=10)
    print("Concluído. 10 jogos gerados.")
    
    # 4. Simulação Monte Carlo no Período de Teste
    print("\nSimulando apostas no período de Teste...")
    ga_results = evaluate_games(ga_games, draws_test)
    random_results = evaluate_games(random_games, draws_test)
    apriori_results = evaluate_games(apriori_games, draws_test)
    
    # 5. Relatório Comparativo
    print("\n================ RESULTADOS DO BACKTESTING ================")
    print(f"{'Métrica':<15} | {'GA (Genético)':<15} | {'Apriori+KMeans':<15} | {'Aposta Cega':<15}")
    print("-" * 65)
    for hits in [3, 4, 5, 6]:
        label = f"{hits} Acertos"
        if hits == 3: label = "Terno (Prox.)"
        elif hits == 4: label = "Quadra"
        elif hits == 5: label = "Quina"
        elif hits == 6: label = "Sena"
        print(f"{label:<15} | {ga_results[hits]:<15} | {apriori_results[hits]:<15} | {random_results[hits]:<15}")
    print("===========================================================\n")
    print("Nota: 'Terno' não paga prêmio, mas indica que o jogo está cobrindo as áreas corretas do espaço de probabilidade.")

    if args.monte_carlo:
        print("\n================ ESTRESSE DE MONTE CARLO (1.000.000 Sorteios) ================")
        print("Gerando 1.000.000 de sorteios aleatórios sintéticos (isso pode levar uns segundos)...")
        # Estratégia ultra-rápida: sortear índices aleatórios de 60 bolas para gerar 1M combinações sem repetição
        mc_draws = np.argsort(np.random.rand(1000000, 60), axis=1)[:, :6] + 1
        
        print("Avaliando bilhetes contra a base sintética vetorizada...")
        def evaluate_monte_carlo(games_list, draws_matrix):
            results = {3: 0, 4: 0, 5: 0, 6: 0}
            for game in games_list:
                # np.isin é vetorizado. Retorna booleano. Somamos o eixo 1 para ter os acertos por sorteio.
                mask = np.isin(draws_matrix, game)
                hits_per_draw = mask.sum(axis=1)
                for k in results.keys():
                    results[k] += np.sum(hits_per_draw == k)
            return results
            
        ga_mc = evaluate_monte_carlo(ga_games, mc_draws)
        rand_mc = evaluate_monte_carlo(random_games, mc_draws)
        apriori_mc = evaluate_monte_carlo(apriori_games, mc_draws)
        
        print(f"{'Métrica':<15} | {'GA (Genético)':<15} | {'Apriori+KMeans':<15} | {'Aposta Cega':<15}")
        print("-" * 65)
        for hits in [3, 4, 5, 6]:
            label = f"{hits} Acertos"
            print(f"{label:<15} | {ga_mc[hits]:<15} | {apriori_mc[hits]:<15} | {rand_mc[hits]:<15}")
        
        # Calcular ROI básico: custo da aposta = 5.0 reais
        # Quadra média = 1000 reais, Quina = 50k, Sena = 50M
        print("\n* P-Value Simplificado: Avaliando se a performance difere da Aleatória em larga escala *")
        print("==============================================================================\n")

if __name__ == "__main__":
    main()
