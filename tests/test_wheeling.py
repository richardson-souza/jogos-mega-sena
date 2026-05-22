import pytest
import itertools
from src.wheeling_systems import generate_greedy_covering

def test_greedy_covering_mathematical_guarantee():
    pool = [2, 3, 4, 8, 11, 25, 27, 28, 37, 39, 40, 43] # Usando 12 para teste rodar rápido
    guarantee = 4
    
    matrix = generate_greedy_covering(pool, ticket_size=6, guarantee=guarantee)
    
    # Validação matemática:
    # Para qualquer combinação de 4 dezenas escolhidas dentro do pool,
    # DEVE existir pelo menos um bilhete gerado que contenha as 4 dezenas.
    
    all_possible_4_hits = list(itertools.combinations(sorted(pool), guarantee))
    
    uncovered = 0
    for hit_4 in all_possible_4_hits:
        hit_set = set(hit_4)
        is_covered = False
        for ticket in matrix:
            if hit_set.issubset(set(ticket)):
                is_covered = True
                break
        if not is_covered:
            uncovered += 1
            
    assert uncovered == 0, f"A matriz falhou em cobrir {uncovered} combinações da garantia!"
    assert len(matrix) > 0

def test_greedy_covering_max_tickets():
    pool = list(range(1, 16)) # 15 números
    max_tickets = 10
    
    matrix = generate_greedy_covering(pool, ticket_size=6, guarantee=4, max_tickets=max_tickets)
    
    assert len(matrix) == max_tickets
    for ticket in matrix:
        assert len(ticket) == 6
