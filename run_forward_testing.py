import pandas as pd
import random
from src.genetic_algorithm import load_historical_games, run_evolution, generate_random_game
from src.association_rules import generate_apriori_kmeans_games
from run_backtesting import evaluate_games

def get_real_future_draws():
    """Retorna os sorteios reais fornecidos pelo usuário para o Forward Testing."""
    return [
        [11, 12, 14, 20, 42, 44], # Concurso 3008 (14/05/2026)
        [4, 6, 8, 18, 21, 30]     # Concurso 3009 (16/05/2026)
    ]

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default="data/processed/mega_sena_features.csv", help='Caminho do dataset')
    args = parser.parse_args()
    filepath = args.filepath
    print("Iniciando Módulo de Forward Testing (Paper Trading) com DADOS REAIS...")
    
    # 1. Carregamento do Treino
    df_train = pd.read_csv(filepath)
    historical_train, freq_train, stats = load_historical_games(filepath)
    print(f"Base de conhecimento isolada. {len(historical_train)} sorteios carregados para treino.")
        
    # 2. Mock do Futuro (Fev a Mai 2026) -> Agora usando Sorteios Reais!
    print("Injetando os sorteios REAIS recentes de Maio/2026 (Concursos 3008 e 3009)...")
    future_draws = get_real_future_draws()
    
    # 3. Geração do Ensemble de Ouro
    print("\n[Motor A] Gerando os 10 melhores bilhetes via Algoritmo Genético...")
    ga_games = run_evolution(
        historical_games=historical_train, 
        frequencies_dict=freq_train, 
        pop_size=1000, 
        generations=500, 
        mutation_rate=0.05,
        stats=stats
    )
    
    print("[Motor B] Gerando os 10 melhores bilhetes via Apriori + K-Means...")
    apriori_games = generate_apriori_kmeans_games(df_train, num_games=10, n_clusters=5)
    
    ensemble_games = ga_games + apriori_games
    print(f"Ensemble montado! Total de {len(ensemble_games)} bilhetes de ouro prontos para produção.")
    
    # 4. Geração de Aposta Cega (Controle)
    random_games = [generate_random_game() for _ in range(20)]
    
    # 5. Simulação Cega
    print("\nIniciando Simulação contra os 30 Sorteios do Futuro...")
    ensemble_results = evaluate_games(ensemble_games, future_draws)
    random_results = evaluate_games(random_games, future_draws)
    
    # 6. Relatório de Produção
    print("\n============== RESULTADOS DO FORWARD TESTING ==============")
    print(f"{'Métrica':<15} | {'ENSEMBLE (20j)':<15} | {'Aposta Cega (20j)':<15}")
    print("-" * 55)
    for hits in [3, 4, 5, 6]:
        label = f"{hits} Acertos"
        if hits == 3: label = "Terno (Prox.)"
        elif hits == 4: label = "Quadra"
        elif hits == 5: label = "Quina"
        elif hits == 6: label = "Sena"
        print(f"{label:<15} | {ensemble_results[hits]:<15} | {random_results[hits]:<15}")
    print("===========================================================\n")
    print("Veredito: Se o Ensemble conseguiu Quadras contra o futuro cego, o modelo está validado.")

if __name__ == "__main__":
    main()
