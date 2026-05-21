import pytest
import pandas as pd
from src.genetic_algorithm import (
    load_historical_games,
    generate_random_game,
    fitness_function,
    crossover,
    mutate,
    run_evolution
)
import os

def test_load_historical_games(tmp_path):
    # Setup fake csv
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
    
    historical = load_historical_games(str(csv_file))
    
    assert len(historical) == 2
    assert frozenset([1, 3, 5, 7, 9, 11]) in historical
    assert frozenset([2, 4, 6, 8, 10, 12]) in historical

def test_generate_random_game():
    game = generate_random_game()
    assert len(game) == 6
    assert all(1 <= x <= 60 for x in game)
    assert len(set(game)) == 6 # No duplicates

def test_fitness_function():
    historical_games = {frozenset([1, 2, 3, 4, 5, 6])}
    
    # Test 1: Historical game penalty (-100)
    game_hist = [1, 2, 3, 4, 5, 6] # sum = 21 (no sum bonus), 3/3 even/odd (bonus +10). Expected: -100 + 10 = -90
    assert fitness_function(game_hist, historical_games) == -90
    
    # Test 2: Perfect game (+10 for Par/Impar, +10 for Sum) -> sum 120-210, not historical
    # e.g., 10, 20, 30, 31, 41, 51 -> sum = 183. Evens: 3, Odds: 3. Expected: +20
    game_perfect = [10, 20, 30, 31, 41, 51]
    assert fitness_function(game_perfect, historical_games) == 20
    
    # Test 3: Bad Par/Impar, Good Sum -> sum = 135. Evens: 6, Odds: 0. Expected: +10
    game_bad_par = [10, 20, 22, 24, 28, 31] # sum 135, evens: 5, odds 1 (wait, I said 6, but 31 is odd). Let's use 32.
    game_bad_par = [10, 20, 22, 24, 28, 32] # sum 136. Evens 6, Odds 0. Expected: +10
    assert fitness_function(game_bad_par, historical_games) == 10
    
    # Test 4: Good Par/Impar, Bad Sum -> sum = 21. Evens: 3, Odds: 3. Expected: +10
    game_bad_sum = [1, 2, 3, 4, 5, 7] # sum = 22. Evens: 2, Odds: 4. (wait, 1,3,5,7=4 odds. 2,4=2 evens). Expected: +10
    assert fitness_function(game_bad_sum, historical_games) == 10

def test_crossover():
    parent1 = [1, 2, 3, 4, 5, 6]
    parent2 = [10, 20, 30, 40, 50, 60]
    
    child = crossover(parent1, parent2)
    
    assert len(child) == 6
    assert len(set(child)) == 6 # No duplicates
    assert all(x in parent1 or x in parent2 for x in child)

def test_mutate():
    game = [1, 2, 3, 4, 5, 6]
    # Com taxa 1.0 (100%), deve mudar pelo menos 1 número, ou retornar o mesmo caso calhe de trocar pelo mesmo (raro, mas evitaremos na implementacao).
    # O importante é que continue com 6 números, únicos e entre 1 e 60.
    mutated = mutate(game, mutation_rate=1.0)
    
    assert len(mutated) == 6
    assert len(set(mutated)) == 6
    assert all(1 <= x <= 60 for x in mutated)
    assert mutated != game # Com 100% de chance, garantimos que muda

def test_run_evolution():
    # Setup dummy historical
    historical = {frozenset([1, 2, 3, 4, 5, 6])}
    
    # Run with small population and generations for quick test
    best_individuals = run_evolution(historical, pop_size=20, generations=10, mutation_rate=0.1)
    
    assert len(best_individuals) == 10
    assert all(len(ind) == 6 for ind in best_individuals)
    assert all(len(set(ind)) == 6 for ind in best_individuals)
    # Check if they are valid ranges
    for ind in best_individuals:
        assert all(1 <= x <= 60 for x in ind)





