import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from src.lstm_model import MegaSenaLSTM, prepare_lstm_data, train_lstm, predict_top_numbers, generate_lstm_games
import numpy as np

def main():
    filepath = "data/processed/mega_sena_features.csv"
    model_path = "lstm_model.pth"
    
    print("Iniciando Módulo de Deep Learning (LSTM)...")
    df = pd.read_csv(filepath)
    
    window_size = 20
    X, y = prepare_lstm_data(df, window_size=window_size)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MegaSenaLSTM()
    
    if os.path.exists(model_path):
        print("Carregando modelo treinado do disco...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("Treinando LSTM do zero. Isso pode demorar alguns minutos...")
        train_lstm(model, dataloader, epochs=50)
        torch.save(model.state_dict(), model_path)
        print("Modelo salvo com sucesso!")
        
    # Extrair os últimos 20 sorteios reais para prever o próximo
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    last_draws = df[cols_bolas].tail(window_size)
    
    # Transformar em one-hot
    one_hot_seq = []
    for _, row in last_draws.iterrows():
        draw = np.zeros(60, dtype=np.float32)
        for val in row.dropna():
            draw[int(val) - 1] = 1.0
        one_hot_seq.append(draw)
        
    last_seq_tensor = torch.tensor(np.array(one_hot_seq))
    
    print("\nAnalisando a série temporal com a LSTM...")
    top_15 = predict_top_numbers(model, last_seq_tensor, top_k=15)
    print(f"🔥 As 15 dezenas mais quentes identificadas: {sorted(top_15)}\n")
    
    print("================ OS 10 BILHETES LSTM ================")
    games = generate_lstm_games(top_15, num_games=10)
    
    for i, game in enumerate(games, 1):
        formatted = " - ".join([f"{n:02d}" for n in game])
        print(f"Bilhete LSTM {i:02d}: [ {formatted} ]")
    print("=====================================================\n")

if __name__ == "__main__":
    main()
