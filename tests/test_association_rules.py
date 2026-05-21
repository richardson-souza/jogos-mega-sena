import pytest
import pandas as pd
from src.association_rules import (
    mine_frequent_itemsets,
    train_kmeans_clusters,
    generate_apriori_kmeans_games
)

def test_mine_frequent_itemsets():
    # Setup dummy data where 1, 2, 3 appear frequently together
    data = {
        'Concurso': [1, 2, 3, 4],
        'Bola1': [1, 1, 1, 10],
        'Bola2': [2, 2, 2, 20],
        'Bola3': [3, 3, 3, 30],
        'Bola4': [4, 5, 6, 40],
        'Bola5': [7, 8, 9, 50],
        'Bola6': [11, 12, 13, 60]
    }
    df = pd.DataFrame(data)
    
    rules = mine_frequent_itemsets(df, min_support=0.5)
    
    # We should have the itemset {1, 2, 3} since it appears in 3 out of 4 games (75% support)
    # The return should be a dataframe with 'itemsets' and 'support' columns.
    assert not rules.empty
    assert 'itemsets' in rules.columns
    assert 'support' in rules.columns
    
    # Check if frozenset({1, 2, 3}) is among the rules
    found = False
    for itemset in rules['itemsets']:
        if frozenset([1, 2, 3]).issubset(itemset):
            found = True
            break
            
    assert found

def test_train_kmeans_clusters():
    # Setup dummy data with heuristic columns
    data = {
        'Soma': [120, 125, 200, 205, 150, 155],
        'Spread': [40, 42, 55, 56, 30, 31],
        'Prop_Pares': [0.5, 0.5, 0.33, 0.33, 0.66, 0.66]
    }
    df = pd.DataFrame(data)
    
    model, centroids = train_kmeans_clusters(df, n_clusters=3)
    
    assert model is not None
    assert len(centroids) == 3
    # Centroids should have 3 dimensions (Soma, Spread, Prop_Pares)
    assert len(centroids[0]) == 3

def test_generate_apriori_kmeans_games():
    # Setup dummy historical features df
    data = {
        'Concurso': [1, 2, 3, 4],
        'Bola1': [1, 1, 1, 10],
        'Bola2': [2, 2, 2, 20],
        'Bola3': [3, 3, 3, 30],
        'Bola4': [4, 5, 6, 40],
        'Bola5': [7, 8, 9, 50],
        'Bola6': [11, 12, 13, 60],
        'Soma': [28, 31, 34, 210],
        'Spread': [10, 11, 12, 50],
        'Prop_Pares': [0.5, 0.5, 0.33, 0.66]
    }
    df = pd.DataFrame(data)
    
    games = generate_apriori_kmeans_games(df, num_games=5, n_clusters=2)
    
    assert len(games) == 5
    for game in games:
        assert len(game) == 6
        assert len(set(game)) == 6
        assert all(1 <= x <= 60 for x in game)


