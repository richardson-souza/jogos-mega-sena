import pandas as pd
import random
import itertools

def load_historical_games(filepath: str) -> set:
    df = pd.read_csv(filepath)
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    historical_set = set()
    for _, row in df[cols_bolas].iterrows():
        historical_set.add(frozenset(row.dropna().astype(int)))
    return historical_set

def generate_random_game() -> list:
    return sorted(random.sample(range(1, 61), 6))

def generate_random_portfolio(size: int = 10) -> list:
    return [generate_random_game() for _ in range(size)]

def individual_game_fitness(game: list, historical_games: set) -> int:
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

def fitness_function(portfolio: list, historical_games: set) -> float:
    # Avaliar o portfolio inteiro somando os fitness individuais
    base_score = sum(individual_game_fitness(game, historical_games) for game in portfolio)
    
    # Calcular Distância de Hamming entre todos os pares (espalhamento máximo)
    # Distância de Hamming adaptada: 6 - tamanho da interseção
    total_distance = 0
    pairs = 0
    for g1, g2 in itertools.combinations(portfolio, 2):
        intersection = len(set(g1).intersection(set(g2)))
        distance = 6 - intersection
        total_distance += distance
        pairs += 1
        
    avg_distance = total_distance / pairs if pairs > 0 else 0
    
    # Adicionar pontuação baseada na distância para favorecer maior cobertura
    # Multiplicador 5.0 para dar relevância razoável em relação aos bônus individuais
    return base_score + (avg_distance * 5.0)

def crossover(parent1: list, parent2: list) -> list:
    # Crossover de portfólios: trocar bilhetes inteiros aleatoriamente
    child = []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(list(g1))
        else:
            child.append(list(g2))
    return child

def mutate(portfolio: list, mutation_rate: float) -> list:
    # Mutação a nível de portfólio: chance de mutar dezenas dentro dos jogos individuais
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

def run_evolution(historical_games: set, pop_size=100, generations=100, mutation_rate=0.05, portfolio_size=10) -> list:
    # População agora é um conjunto de Portfólios
    population = [generate_random_portfolio(portfolio_size) for _ in range(pop_size)]
    
    for gen in range(generations):
        # Avaliar fitness
        scored_pop = [(port, fitness_function(port, historical_games)) for port in population]
        # Ordenar decrescente (maior fitness primeiro)
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar os melhores 20%
        top_20_percent_idx = max(1, int(pop_size * 0.2))
        best_individuals = [x[0] for x in scored_pop[:top_20_percent_idx]]
        
        # Nova população inicia com os melhores (Elitismo)
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
    scored_pop = [(port, fitness_function(port, historical_games)) for port in population]
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    
    # Extrair e retornar apenas o MELHOR portfólio absoluto
    best_portfolio = scored_pop[0][0]
    return best_portfolio
