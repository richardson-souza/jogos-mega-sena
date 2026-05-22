import itertools

def generate_greedy_covering(pool: list, ticket_size: int = 6, guarantee: int = 4, max_tickets: int = None) -> list:
    """
    Gera uma matriz ortogonal (wheeling system) usando algoritmo guloso (Greedy Set Cover).
    Garante que se 'guarantee' dezenas do pool forem sorteadas, haverá pelo menos um bilhete
    contendo essas 'guarantee' dezenas.
    
    Args:
        pool (list): Lista de dezenas selecionadas (ex: as 15 dezenas mais quentes).
        ticket_size (int): Tamanho de cada bilhete gerado (padrão 6).
        guarantee (int): Condição de garantia de acerto (padrão 4 = Quadra).
        max_tickets (int): (Opcional) Limite de bilhetes a gerar. Se omitido, gera a matriz completa.
        
    Returns:
        list: Lista de bilhetes (listas de inteiros ordenados).
    """
    if len(pool) < ticket_size:
        raise ValueError("O pool deve ter tamanho maior ou igual ao tamanho do bilhete.")
        
    pool_sorted = sorted(pool)
    
    # Todos os subconjuntos de tamanho 'guarantee' que precisamos cobrir
    targets = set(itertools.combinations(pool_sorted, guarantee))
    uncovered_targets = set(targets)
    
    # Todas as possíveis apostas (candidatos) a serem testadas
    candidates = list(itertools.combinations(pool_sorted, ticket_size))
    
    tickets = []
    
    while uncovered_targets:
        best_candidate = None
        best_coverage = []
        
        # Encontrar o candidato que cobre o maior número de targets não cobertos
        for candidate in candidates:
            # Targets que este candidato cobre:
            # (todas as combinações de tamanho 'guarantee' dentro deste candidato)
            covers = set(itertools.combinations(candidate, guarantee))
            covers_uncovered = covers.intersection(uncovered_targets)
            
            if len(covers_uncovered) > len(best_coverage):
                best_coverage = covers_uncovered
                best_candidate = candidate
                
        if not best_candidate:
            break
            
        tickets.append(list(best_candidate))
        uncovered_targets -= best_coverage
        
        if max_tickets and len(tickets) >= max_tickets:
            break
            
    return tickets
