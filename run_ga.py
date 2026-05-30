import os
import argparse
from src.genetic_algorithm import load_historical_games, run_evolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default="data/processed/mega_sena_features.csv", help='Caminho do dataset')
    args = parser.parse_args()
    filepath = args.filepath
    if not os.path.exists(filepath):
        print(f"Erro: O arquivo {filepath} não existe.")
        print("Execute 'python run.py' primeiro para baixar e processar os dados históricos.")
        return
        
    print(f"Carregando histórico de {filepath}...")
    historical_games, frequencies, stats = load_historical_games(filepath)
    print(f"{len(historical_games)} jogos carregados.")
    
    print("Iniciando algoritmo genético. Isso pode demorar um pouco...")
    # Usaremos 500 gerações e 1000 indivíduos conforme abordagem.
    best_games = run_evolution(
        historical_games=historical_games, 
        frequencies_dict=frequencies,
        pop_size=1000, 
        generations=500, 
        mutation_rate=0.05,
        stats=stats
    )
    
    print("\n========= TOP 10 JOGOS GERADOS =========")
    for i, game in enumerate(best_games, 1):
        soma = sum(game)
        pares = sum(1 for x in game if x % 2 == 0)
        impares = 6 - pares
        print(f"#{i:02d}: {game} | Soma: {soma} | Pares: {pares} | Ímpares: {impares}")
        
if __name__ == "__main__":
    main()
