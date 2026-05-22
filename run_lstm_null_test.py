import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.lstm_model import MegaSenaLSTM, prepare_lstm_data, train_lstm

def generate_fake_data(reference_df):
    """Gera um DataFrame com ruído puro, mas com o mesmo formato do real."""
    fake_df = reference_df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    # Substituir os valores reais por sorteios uniformemente aleatórios (ruído)
    fake_draws = np.argsort(np.random.rand(len(fake_df), 60), axis=1)[:, :6] + 1
    fake_df[cols_bolas] = fake_draws
    
    return fake_df

def main():
    print("================ TESTE DE HIPÓTESE NULA (LSTM) ================")
    
    # 1. Carregar Dados Reais e Gerar Fakes
    filepath = "data/processed/mega_sena_features.csv"
    real_df = pd.read_csv(filepath)
    fake_df = generate_fake_data(real_df)
    
    window_size = 20
    print("Preparando tensores (Real e Fake)...")
    X_real, y_real = prepare_lstm_data(real_df, window_size=window_size)
    X_fake, y_fake = prepare_lstm_data(fake_df, window_size=window_size)
    
    real_loader = DataLoader(TensorDataset(X_real, y_real), batch_size=32, shuffle=True)
    fake_loader = DataLoader(TensorDataset(X_fake, y_fake), batch_size=32, shuffle=True)
    
    # 2. Treinamento
    epochs = 30
    print(f"Treinando modelo REAL por {epochs} épocas...")
    model_real = MegaSenaLSTM()
    losses_real = train_lstm(model_real, real_loader, epochs=epochs)
    
    print(f"Treinando modelo FAKE por {epochs} épocas...")
    model_fake = MegaSenaLSTM()
    losses_fake = train_lstm(model_fake, fake_loader, epochs=epochs)
    
    # 3. Análise da Hipótese Nula
    print("\n================ RESULTADOS DA HIPÓTESE NULA ================")
    print(f"Loss Real (Inicial -> Final): {losses_real[0]:.4f} -> {losses_real[-1]:.4f}")
    print(f"Loss Fake (Inicial -> Final): {losses_fake[0]:.4f} -> {losses_fake[-1]:.4f}")
    
    drop_real = losses_real[0] - losses_real[-1]
    drop_fake = losses_fake[0] - losses_fake[-1]
    
    print(f"Queda de Loss (Real): {drop_real:.4f}")
    print(f"Queda de Loss (Fake): {drop_fake:.4f}")
    
    if drop_fake > (drop_real * 0.8):
        print("\n[!] ALERTA CRÍTICO: A rede apresentou uma queda de erro significativa no dataset")
        print("    de ruído puro (aleatório). Isso significa que o modelo tem capacidade excessiva")
        print("    (overfitting) e está memorizando a aleatoriedade ao invés de aprender padrões")
        print("    generalizáveis temporais. Considere adicionar Dropout mais forte ou reduzir a")
        print("    complexidade (hidden_size) da LSTM.")
    else:
        print("\n[OK] SUCESSO: A rede não conseguiu memorizar os dados puramente aleatórios tão bem")
        print("     quanto os dados reais. Isso indica que há algum padrão real sendo aprendido")
        print("     (o que rejeita a hipótese nula de que os sorteios são puramente um 'passeio aleatório').")
    print("=============================================================\n")

if __name__ == "__main__":
    main()
