import pytest
import pandas as pd
from src.genetic_algorithm import (
    load_historical_games,
    generate_random_game,
    generate_random_portfolio,
    generate_boosted_portfolio,
    individual_game_fitness,
    fitness_function,
    crossover,
    mutate,
    run_evolution
)

def test_load_historical_games(tmp_path):
    csv_file = tmp_path / "fake_features.csv"
    data = {
        'Concurso': [1, 2],
        'Bola1': [1, 2],
        'Bola2': [3, 4],
        'Bola3': [5, 6],
        'Bola4': [7, 8],
        'Bola5': [9, 10],
        'Bola6': [11, 12]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    historical, frequencies = load_historical_games(str(csv_file))
    assert len(historical) == 2
    assert frequencies[1] == 0.5

def test_generate_random_portfolio():
    portfolio = generate_random_portfolio(10)
    assert len(portfolio) == 10
    assert len(portfolio[0]) == 6

def test_generate_boosted_portfolio():
    itemsets = [[10, 20], [30, 40, 50]]
    port = generate_boosted_portfolio(10, itemsets)
    assert len(port) == 10
    assert len(port[0]) == 6
    assert len(set(port[0])) == 6

def test_individual_game_fitness_standard():
    historical_games = {frozenset([1, 2, 3, 4, 5, 6])}
    freq_dict = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1}
    game_hist = [1, 2, 3, 4, 5, 6]
    assert individual_game_fitness(game_hist, historical_games, freq_dict) == -84
    game_perfect = [10, 20, 30, 31, 41, 51]
    assert individual_game_fitness(game_perfect, historical_games, freq_dict) == 20

def test_individual_game_fitness_stacking():
    historical_games = set()
    freq_dict = {}
    itemsets = [[10, 20]]
    centroids = [[150, 30, 0.5]] # soma=150, spread=30, pares=0.5
    
    # game 1 atende a regra do apriori (tem 10 e 20)
    game = [10, 20, 25, 27, 28, 40] 
    
    score_standard = individual_game_fitness(game, historical_games, freq_dict, method='standard')
    score_stacking = individual_game_fitness(game, historical_games, freq_dict, method='stacking', itemsets=itemsets, centroids=centroids)
    
    # O stacking deve diferir do standard (bônus apriori len(2)*5 = 10, menos penalidade kmeans)
    assert score_stacking != score_standard

def test_fitness_function_portfolio_spread():
    historical_games = set()
    freq_dict = {}
    game = [10, 20, 30, 31, 41, 51]
    port_identical = [game, game]
    game2 = [2, 4, 6, 9, 11, 13]
    port_different = [game, game2]
    
    fit_identical = fitness_function(port_identical, historical_games, freq_dict)
    fit_different = fitness_function(port_different, historical_games, freq_dict)
    assert fit_different > fit_identical

def test_run_evolution_portfolio():
    historical = {frozenset([1, 2, 3, 4, 5, 6])}
    freq = {}
    # Run standard
    best_portfolio = run_evolution(historical, freq, pop_size=10, generations=2, mutation_rate=0.1, portfolio_size=2)
    assert len(best_portfolio) == 2
    
    # Run boosted
    itemsets = [[1, 2], [3, 4]]
    best_boosted = run_evolution(historical, freq, pop_size=10, generations=2, method='boosted', itemsets=itemsets, portfolio_size=2)
    assert len(best_boosted) == 2

def test_individual_game_fitness_custom_weights():
    historical_games = set()
    freq_dict = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1}
    itemsets = [[1, 2]]
    centroids = [[21, 5, 0.5]]
    
    game = [1, 2, 3, 4, 5, 6]
    
    weights_custom = {'w_freq': 1.0, 'w_apriori': 100.0, 'w_kmeans': 0.1, 'w_hamming': 1.0}
    score_custom = individual_game_fitness(game, historical_games, freq_dict, method='stacking', itemsets=itemsets, centroids=centroids, weights_dict=weights_custom)
    
    weights_zero = {'w_freq': 1.0, 'w_apriori': 0.0, 'w_kmeans': 0.1, 'w_hamming': 1.0}
    score_zero = individual_game_fitness(game, historical_games, freq_dict, method='stacking', itemsets=itemsets, centroids=centroids, weights_dict=weights_zero)
    
    assert score_custom > score_zero
    assert score_custom - score_zero == 200.0 # len([1, 2]) * 100.0
