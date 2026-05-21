import pandas as pd
import random
import numpy as np
from mlxtend.frequent_patterns import fpgrowth

def mine_frequent_itemsets(df: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    # Criar lista de listas
    transactions = []
    for _, row in df[cols_bolas].iterrows():
        transactions.append([int(x) for x in row.dropna()])
        
    # One-hot encoding manual para dezenas de 1 a 60
    encoded_data = []
    for t in transactions:
        encoded_data.append({i: (i in t) for i in range(1, 61)})
        
    df_encoded = pd.DataFrame(encoded_data)
    
    # Mining
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    return frequent_itemsets

from sklearn.cluster import KMeans

def train_kmeans_clusters(df: pd.DataFrame, n_clusters: int = 5):
    features = ['Soma', 'Spread', 'Prop_Pares']
    X = df[features].copy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)
    
    return kmeans, kmeans.cluster_centers_

def generate_apriori_kmeans_games(df: pd.DataFrame, num_games: int = 10, n_clusters: int = 5) -> list:
    # 1. Obter regras (itemsets) e filtrar para tamanho >= 2
    itemsets_df = mine_frequent_itemsets(df, min_support=0.01)
    if not itemsets_df.empty:
        itemsets_df['length'] = itemsets_df['itemsets'].apply(lambda x: len(x))
        valid_itemsets = itemsets_df[itemsets_df['length'] >= 2]['itemsets'].tolist()
    else:
        valid_itemsets = []
        
    # 2. Treinar K-Means
    kmeans, centroids = train_kmeans_clusters(df, n_clusters=n_clusters)
    
    generated_games = []
    
    for _ in range(num_games):
        # Escolher um perfil alvo (centroide)
        target_centroid = random.choice(centroids) # [Soma, Spread, Prop_Pares]
        
        # Escolher uma semente do Apriori
        seed = set(random.choice(valid_itemsets)) if valid_itemsets else set()
        
        best_game = None
        best_dist = float('inf')
        
        # Gerar 100 candidatos aleatórios a partir da semente e escolher o mais próximo do perfil
        for _ in range(100):
            candidate = set(seed)
            available = list(set(range(1, 61)) - candidate)
            # Completar o jogo
            candidate.update(random.sample(available, 6 - len(candidate)))
            candidate_list = sorted(list(candidate))
            
            # Calcular heurísticas do candidato
            c_soma = sum(candidate_list)
            c_spread = max(candidate_list) - min(candidate_list)
            c_prop_pares = sum(1 for x in candidate_list if x % 2 == 0) / 6.0
            
            # Distância Euclidiana (simples, sem normalizar para manter leve)
            dist = np.sqrt((c_soma - target_centroid[0])**2 + 
                           (c_spread - target_centroid[1])**2 + 
                           ((c_prop_pares - target_centroid[2])*100)**2) # peso maior na prop
            
            if dist < best_dist:
                best_dist = dist
                best_game = candidate_list
                
        generated_games.append(best_game)
        
    return generated_games
