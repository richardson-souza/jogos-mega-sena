import pandas as pd
import random

def load_historical_games(filepath: str) -> set:
    df = pd.read_csv(filepath)
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_set = set()
    for _, row in df[cols_bolas].iterrows():
        historical_set.add(frozenset(row.dropna().astype(int)))
    return historical_set

def generate_random_game() -> list:
    return sorted(random.sample(range(1, 61), 6))

def fitness_function(game: list, historical_games: set) -> int:
    score = 0
    
    # Par/Ímpar: 3/3, 4/2, 2/4
    evens = sum(1 for x in game if x % 2 == 0)
    odds = 6 - evens
    if (evens == 3 and odds == 3) or (evens == 4 and odds == 2) or (evens == 2 and odds == 4):
        score += 10
        
    # Soma entre 120 e 210
    total_sum = sum(game)
    if 120 <= total_sum <= 210:
        score += 10
        
    # Jogo histórico
    if frozenset(game) in historical_games:
        score -= 100
        
    return score

def crossover(parent1: list, parent2: list) -> list:
    # Combinar pais, remover duplicatas
    pool = list(set(parent1 + parent2))
    # Selecionar 6 números aleatórios
    child = random.sample(pool, 6)
    return sorted(child)

def mutate(game: list, mutation_rate: float) -> list:
    game_copy = list(game)
    if random.random() < mutation_rate:
        idx_to_replace = random.randint(0, 5)
        current_set = set(game_copy)
        available_numbers = list(set(range(1, 61)) - current_set)
        if available_numbers:
            new_number = random.choice(available_numbers)
            game_copy[idx_to_replace] = new_number
    return sorted(game_copy)

def run_evolution(historical_games: set, pop_size=1000, generations=500, mutation_rate=0.05) -> list:
    population = [generate_random_game() for _ in range(pop_size)]
    
    for gen in range(generations):
        # Avaliar fitness
        scored_pop = [(game, fitness_function(game, historical_games)) for game in population]
        # Ordenar decrescente (maior fitness primeiro)
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar os melhores 20%
        top_20_percent_idx = max(1, int(pop_size * 0.2))
        best_individuals = [x[0] for x in scored_pop[:top_20_percent_idx]]
        
        # Nova população inicia com os melhores
        new_population = best_individuals.copy()
        
        # Completar o resto com crossover e mutação
        while len(new_population) < pop_size:
            parent1 = random.choice(best_individuals)
            parent2 = random.choice(best_individuals)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
            
        population = new_population
        
    # Ultima avaliação
    scored_pop = [(game, fitness_function(game, historical_games)) for game in population]
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    
    # Extrair os 10 melhores da última geração
    final_best = [x[0] for x in scored_pop[:10]]
    return final_best
