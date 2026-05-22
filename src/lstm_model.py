import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def prepare_lstm_data(df: pd.DataFrame, window_size: int = 20):
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    # Criar sequências one-hot
    one_hot_sequences = []
    for _, row in df[cols_bolas].iterrows():
        draw = np.zeros(60, dtype=np.float32)
        for val in row.dropna():
            draw[int(val) - 1] = 1.0 # 0-indexed
        one_hot_sequences.append(draw)
        
    X, y = [], []
    for i in range(len(one_hot_sequences) - window_size):
        X.append(one_hot_sequences[i : i + window_size])
        y.append(one_hot_sequences[i + window_size])
        
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

class MegaSenaLSTM(nn.Module):
    def __init__(self, input_size=60, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 60)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x is (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Extract the last time step output for each sequence in the batch
        last_out = out[:, -1, :] 
        return self.sigmoid(self.fc(last_out))

def train_lstm(model, dataloader, epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
    return losses
            
def predict_top_numbers(model, last_sequence, top_k=15):
    model.eval()
    with torch.no_grad():
        # last_sequence shape should be (1, window_size, 60)
        if len(last_sequence.shape) == 2:
            last_sequence = last_sequence.unsqueeze(0)
            
        probs = model(last_sequence).squeeze(0) # shape (60)
        
        # Get top k indices
        top_indices = torch.argsort(probs, descending=True)[:top_k]
        
        # Convert indices to numbers (1 to 60)
        top_numbers = [idx.item() + 1 for idx in top_indices]
        
    return top_numbers

from src.wheeling_systems import generate_greedy_covering

def generate_lstm_games(top_numbers: list, num_games: int = 10):
    # Usar Wheeling System determinístico (Algoritmo Guloso)
    # Garante cobertura matemática (Quadra) focada nas dezenas mais quentes.
    return generate_greedy_covering(top_numbers, ticket_size=6, guarantee=4, max_tickets=num_games)
