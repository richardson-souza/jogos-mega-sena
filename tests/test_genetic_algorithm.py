import pytest
import pandas as pd
from src.genetic_algorithm import (
    load_historical_games,
    generate_random_game,
    generate_random_portfolio,
    individual_game_fitness,
    fitness_function,
    crossover,
    mutate,
    run_evolution
)
import os

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
    assert frozenset([1, 3, 5, 7, 9, 11]) in historical
    
    # Cada número apareceu exatamente 1 vez em 2 sorteios. Freq = 0.5
    assert frequencies[1] == 0.5
    assert frequencies[12] == 0.5

def test_generate_random_portfolio():
    portfolio = generate_random_portfolio(10)
    assert len(portfolio) == 10
    assert len(portfolio[0]) == 6
    assert len(set(portfolio[0])) == 6

def test_individual_game_fitness():
    historical_games = {frozenset([1, 2, 3, 4, 5, 6])}
    freq_dict = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1} # Soma freqs: 0.6 * 10 = 6
    
    game_hist = [1, 2, 3, 4, 5, 6] # sum = 21, 3/3 par/impar -> -100 + 10 + 6 = -84
    assert individual_game_fitness(game_hist, historical_games, freq_dict) == -84
    
    game_perfect = [10, 20, 30, 31, 41, 51] # sum=183, 3/3 par/impar -> 20 + 0 = 20
    assert individual_game_fitness(game_perfect, historical_games, freq_dict) == 20

def test_fitness_function_portfolio_spread():
    historical_games = set()
    freq_dict = {}
    # Portfolio com jogos idênticos (zero espalhamento)
    game = [10, 20, 30, 31, 41, 51] # fitness ind: 20
    port_identical = [game, game] # fitness individual soma: 40. Distância: 0. Total: 40
    
    # Portfolio com jogos totalmente diferentes (espalhamento maximo de 6)
    game2 = [2, 4, 6, 9, 11, 13] # sum=45 (0 bonus), 3/3 par/impar (+10). fitness ind: 10
    # Soma = 20 + 10 = 30. Distancia: 6. Multiplicador: 6 * 5.0 = 30. Total: 30 + 30 = 60
    port_different = [game, game2]
    
    fit_identical = fitness_function(port_identical, historical_games, freq_dict)
    fit_different = fitness_function(port_different, historical_games, freq_dict)
    
    assert fit_identical == 40
    assert fit_different == 60 # Confirma que o espalhamento aumenta o score mesmo com jogos individualmente "piores"

def test_crossover_portfolio():
    p1 = [[1,2,3,4,5,6], [7,8,9,10,11,12]]
    p2 = [[13,14,15,16,17,18], [19,20,21,22,23,24]]
    
    child = crossover(p1, p2)
    assert len(child) == 2
    assert child[0] in [p1[0], p2[0]]
    assert child[1] in [p1[1], p2[1]]

def test_mutate_portfolio():
    port = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]]
    mutated = mutate(port, mutation_rate=1.0)
    
    assert len(mutated) == 2
    assert mutated[0] != port[0]
    assert mutated[1] != port[1]
    assert len(set(mutated[0])) == 6

def test_run_evolution_portfolio():
    historical = {frozenset([1, 2, 3, 4, 5, 6])}
    freq = {}
    best_portfolio = run_evolution(historical, freq, pop_size=10, generations=5, mutation_rate=0.1, portfolio_size=5)
    
    assert len(best_portfolio) == 5
    assert all(len(ind) == 6 for ind in best_portfolio)
