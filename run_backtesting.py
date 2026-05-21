import pandas as pd
import random
from datetime import timedelta
from src.genetic_algorithm import load_historical_games, run_evolution, generate_random_game

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
    
    # Criar set histórico de treino para o GA
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_train = set()
    for _, row in df_train[cols_bolas].iterrows():
        historical_train.add(frozenset(row.dropna().astype(int)))
        
    # Extrair lista de sorteios reais do teste
    draws_test = []
    for _, row in df_test[cols_bolas].iterrows():
        draws_test.append(row.dropna().astype(int).tolist())
        
    # 2. Gerar Jogos Otimizados via GA
    print("\nExecutando Algoritmo Genético (apenas no conjunto de treino)...")
    ga_games = run_evolution(historical_train, pop_size=1000, generations=500, mutation_rate=0.05)
    print("Concluído. 10 jogos gerados.")
    
    # 3. Gerar Jogos Aleatórios (Aposta Cega)
    random_games = [generate_random_game() for _ in range(10)]
    
    # 4. Simulação Monte Carlo no Período de Teste
    print("\nSimulando apostas no período de Teste...")
    ga_results = evaluate_games(ga_games, draws_test)
    random_results = evaluate_games(random_games, draws_test)
    
    # 5. Relatório Comparativo
    print("\n================ RESULTADOS DO BACKTESTING ================")
    print(f"{'Métrica':<15} | {'GA (Otimizado)':<15} | {'Aposta Cega':<15}")
    print("-" * 50)
    for hits in [3, 4, 5, 6]:
        label = f"{hits} Acertos"
        if hits == 3: label = "Terno (Prox.)"
        elif hits == 4: label = "Quadra"
        elif hits == 5: label = "Quina"
        elif hits == 6: label = "Sena"
        print(f"{label:<15} | {ga_results[hits]:<15} | {random_results[hits]:<15}")
    print("===========================================================\n")
    print("Nota: 'Terno' não paga prêmio, mas indica que o jogo está cobrindo as áreas corretas do espaço de probabilidade.")

if __name__ == "__main__":
    main()
