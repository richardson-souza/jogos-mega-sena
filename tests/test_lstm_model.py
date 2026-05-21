import pytest
import pandas as pd
import torch
from src.lstm_model import prepare_lstm_data

def test_prepare_lstm_data():
    # Setup dummy data with 5 draws
    data = {
        'Concurso': [1, 2, 3, 4, 5],
        'Bola1': [1, 2, 3, 4, 5],
        'Bola2': [10, 20, 30, 40, 50],
        'Bola3': [11, 21, 31, 41, 51],
        'Bola4': [12, 22, 32, 42, 52],
        'Bola5': [13, 23, 33, 43, 53],
        'Bola6': [14, 24, 34, 44, 54]
    }
    df = pd.DataFrame(data)
    
    # We want a window size of 2
    # So sequence 1: [draw 1, draw 2] -> target: draw 3
    # Sequence 2: [draw 2, draw 3] -> target: draw 4
    # Sequence 3: [draw 3, draw 4] -> target: draw 5
    
    X, y = prepare_lstm_data(df, window_size=2)
    
    assert X is not None
    assert y is not None
    
    # X shape should be (3, 2, 60) -> 3 sequences, window of 2, 60 features
    assert X.shape == (3, 2, 60)
    # y shape should be (3, 60)
    assert y.shape == (3, 60)
    
    # Check if One-Hot is correct for first draw in first sequence (draw 1)
    # Draw 1 has balls: 1, 10, 11, 12, 13, 14
    # Note: 0-indexed tensor means ball 1 is at index 0
    assert X[0, 0, 0] == 1.0 # Ball 1
    assert X[0, 0, 9] == 1.0 # Ball 10
    assert X[0, 0, 10] == 1.0 # Ball 11
    assert X[0, 0, 59] == 0.0 # Ball 60 should be 0

def test_megasena_lstm_architecture():
    from src.lstm_model import MegaSenaLSTM
    model = MegaSenaLSTM()
    
    # Batch size 3, Window size 20, Features 60
    dummy_input = torch.rand((3, 20, 60))
    
    # Forward pass
    output = model(dummy_input)
    
    # Should output exactly (batch_size, 60) for predicting the 60 numbers
    assert output.shape == (3, 60)
    
    # Should be probabilities (Sigmoid activation)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

def test_train_predict_generate():
    from src.lstm_model import MegaSenaLSTM, train_lstm, predict_top_numbers, generate_lstm_games
    from torch.utils.data import TensorDataset, DataLoader
    
    model = MegaSenaLSTM(hidden_size=32, num_layers=1, dropout=0.0)
    
    # Mock data
    X = torch.rand((10, 20, 60))
    y = torch.rand((10, 60))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Train for 1 epoch
    train_lstm(model, dataloader, epochs=1)
    
    # Predict
    last_seq = torch.rand((1, 20, 60))
    top_15 = predict_top_numbers(model, last_seq, top_k=15)
    
    assert len(top_15) == 15
    assert len(set(top_15)) == 15
    assert all(1 <= x <= 60 for x in top_15)
    
    # Generate
    games = generate_lstm_games(top_15, num_games=5)
    assert len(games) == 5
    for game in games:
        assert len(game) == 6
        assert len(set(game)) == 6
        assert all(x in top_15 for x in game)


