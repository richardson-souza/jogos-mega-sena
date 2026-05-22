import pandas as pd
import random
import itertools
import math

def load_historical_games(filepath: str) -> tuple[set, dict]:
    df = pd.read_csv(filepath)
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_set = set()
    
    # Calcular frequências relativas (proporção de sorteios em que a dezena apareceu)
    all_numbers = df[cols_bolas].values.flatten()
    all_numbers = all_numbers[~pd.isna(all_numbers)].astype(int)
    counts = pd.Series(all_numbers).value_counts()
    
    # Frequência relativa: ocorrências / total de sorteios
    frequencies_dict = (counts / len(df)).to_dict()
    
    for _, row in df[cols_bolas].iterrows():
        historical_set.add(frozenset(row.dropna().astype(int)))
        
    return historical_set, frequencies_dict

def generate_random_game() -> list:
    return sorted(random.sample(range(1, 61), 6))

def generate_random_portfolio(size: int = 10) -> list:
    return [generate_random_game() for _ in range(size)]

def generate_boosted_portfolio(size: int, itemsets: list) -> list:
    """Gera um portfólio usando itemsets do Apriori como sementes de alta probabilidade."""
    portfolio = []
    for _ in range(size):
        if itemsets:
            seed = set(random.choice(itemsets))
        else:
            seed = set()
            
        available_numbers = list(set(range(1, 61)) - seed)
        needed = 6 - len(seed)
        
        fill = random.sample(available_numbers, needed) if needed > 0 else []
        game = sorted(list(seed) + fill)
        portfolio.append(game)
    return portfolio

def individual_game_fitness(game: list, historical_games: set, frequencies_dict: dict, method='standard', itemsets=None, centroids=None) -> float:
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
        
    # Bônus de Frequência Histórica
    freq_sum = sum(frequencies_dict.get(num, 0) for num in game)
    score += freq_sum * 10.0
    
    # Lógica de Ensemble: Stacking
    if method == 'stacking':
        # Recompensa Apriori: +5.0 pontos por cada número envolvido numa regra atendida
        apriori_score = 0
        if itemsets:
            for iset in itemsets:
                if set(iset).issubset(set(game)):
                    apriori_score += len(iset) * 5.0
                    
        # Penalidade KMeans: Distância ao centroide mais próximo
        kmeans_score = 0
        if centroids is not None and len(centroids) > 0:
            c_soma = total_sum
            c_spread = max(game) - min(game)
            c_prop_pares = evens / 6.0
            
            min_dist = float('inf')
            for centroid in centroids:
                # Centroide: [Soma, Spread, Prop_Pares]
                dist = math.sqrt((c_soma - centroid[0])**2 + (c_spread - centroid[1])**2 + ((c_prop_pares - centroid[2])*100)**2)
                if dist < min_dist:
                    min_dist = dist
                    
            # Penalidade suave proporcional à distância da regra latente
            kmeans_score = - (min_dist * 0.5) 
            
        score += apriori_score + kmeans_score
        
    return score

def fitness_function(portfolio: list, historical_games: set, frequencies_dict: dict, method='standard', itemsets=None, centroids=None) -> float:
    # Avaliar o portfolio inteiro somando os fitness individuais
    base_score = sum(individual_game_fitness(game, historical_games, frequencies_dict, method, itemsets, centroids) for game in portfolio)
    
    # Calcular Distância de Hamming entre todos os pares (espalhamento máximo)
    total_distance = 0
    pairs = 0
    for g1, g2 in itertools.combinations(portfolio, 2):
        intersection = len(set(g1).intersection(set(g2)))
        distance = 6 - intersection
        total_distance += distance
        pairs += 1
        
    avg_distance = total_distance / pairs if pairs > 0 else 0
    return base_score + (avg_distance * 5.0)

def crossover(parent1: list, parent2: list) -> list:
    child = []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(list(g1))
        else:
            child.append(list(g2))
    return child

def mutate(portfolio: list, mutation_rate: float) -> list:
    new_portfolio = []
    for game in portfolio:
        game_copy = list(game)
        if random.random() < mutation_rate:
            idx_to_replace = random.randint(0, 5)
            current_set = set(game_copy)
            available_numbers = list(set(range(1, 61)) - current_set)
            if available_numbers:
                new_number = random.choice(available_numbers)
                game_copy[idx_to_replace] = new_number
        new_portfolio.append(sorted(game_copy))
    return new_portfolio

def run_evolution(historical_games: set, frequencies_dict: dict, pop_size=100, generations=100, mutation_rate=0.05, portfolio_size=10, method='standard', itemsets=None, centroids=None) -> list:
    
    # Inicialização da População
    if method == 'boosted':
        population = [generate_boosted_portfolio(portfolio_size, itemsets) for _ in range(pop_size)]
    else:
        population = [generate_random_portfolio(portfolio_size) for _ in range(pop_size)]
    
    for gen in range(generations):
        # Avaliar fitness
        scored_pop = [(port, fitness_function(port, historical_games, frequencies_dict, method, itemsets, centroids)) for port in population]
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar os melhores 20%
        top_20_percent_idx = max(1, int(pop_size * 0.2))
        best_individuals = [x[0] for x in scored_pop[:top_20_percent_idx]]
        
        # Elitismo
        new_population = best_individuals.copy()
        
        # Crossover e mutação
        while len(new_population) < pop_size:
            parent1 = random.choice(best_individuals)
            parent2 = random.choice(best_individuals)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
            
        population = new_population
        
    # Ultima avaliação
    scored_pop = [(port, fitness_function(port, historical_games, frequencies_dict, method, itemsets, centroids)) for port in population]
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    
    best_portfolio = scored_pop[0][0]
    return best_portfolio
