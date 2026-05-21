import pandas as pd
import pytest
from src.etl import load_and_clean_data, feature_sum, feature_even_odd, feature_spread, feature_delay_matrix, download_data
from unittest.mock import patch, Mock

def test_load_and_clean_data():
    # Sample data with a null row, a duplicate row, and unordered
    data = {
        'Concurso': [3, 1, 1, 2, 4],
        'Data do Sorteio': ['25/03/1996', '11/03/1996', '11/03/1996', '18/03/1996', '01/04/1996'],
        'Bola1': [10, 4, 4, 9, 1],
        'Bola2': [11, 5, 5, 37, 5],
        'Bola3': [29, 30, 30, 39, 6],
        'Bola4': [30, 33, 33, 41, 27],
        'Bola5': [36, 41, 41, 43, 42],
        'Bola6': [47, 52, 52, 49, None], # One null value
        'OutraColuna': ['A', 'B', 'B', 'C', 'D']
    }
    df = pd.DataFrame(data)
    
    cleaned_df = load_and_clean_data(df)
    
    # 1. Isolate columns
    expected_columns = ['Concurso', 'Data do Sorteio', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    assert list(cleaned_df.columns) == expected_columns
    
    # 2. Remove nulls (Concurso 4 has None in Bola6)
    # 3. Remove duplicates (Concurso 1 is duplicated)
    # So we expect Concursos: 1, 2, 3
    assert len(cleaned_df) == 3
    
    # 4. Order chronologically by Concurso
    assert list(cleaned_df['Concurso']) == [1, 2, 3]

def test_feature_sum():
    data = {
        'Concurso': [1],
        'Bola1': [1],
        'Bola2': [2],
        'Bola3': [3],
        'Bola4': [4],
        'Bola5': [5],
        'Bola6': [6]
    }
    df = pd.DataFrame(data)
    result_df = feature_sum(df)
    
    assert 'Soma' in result_df.columns
    assert result_df['Soma'].iloc[0] == 21

def test_feature_even_odd():
    data = {
        'Concurso': [1],
        'Bola1': [2],
        'Bola2': [4],
        'Bola3': [6],
        'Bola4': [7],
        'Bola5': [9],
        'Bola6': [11]
    }
    df = pd.DataFrame(data)
    result_df = feature_even_odd(df)
    
    assert 'Qtd_Pares' in result_df.columns
    assert 'Qtd_Impares' in result_df.columns
    assert 'Prop_Pares' in result_df.columns
    
    assert result_df['Qtd_Pares'].iloc[0] == 3
    assert result_df['Qtd_Impares'].iloc[0] == 3
    assert result_df['Prop_Pares'].iloc[0] == 0.5

def test_feature_spread():
    data = {
        'Concurso': [1],
        'Bola1': [5],
        'Bola2': [15],
        'Bola3': [25],
        'Bola4': [35],
        'Bola5': [45],
        'Bola6': [55] # Spread should be 55 - 5 = 50
    }
    df = pd.DataFrame(data)
    result_df = feature_spread(df)
    
    assert 'Spread' in result_df.columns
    assert result_df['Spread'].iloc[0] == 50

def test_feature_delay_matrix():
    data = {
        'Concurso': [1, 2],
        'Bola1': [1, 7],
        'Bola2': [2, 8],
        'Bola3': [3, 9],
        'Bola4': [4, 10],
        'Bola5': [5, 11],
        'Bola6': [6, 12]
    }
    df = pd.DataFrame(data)
    result_df = feature_delay_matrix(df)
    
    assert 'Atraso_1' in result_df.columns
    assert 'Atraso_60' in result_df.columns
    
    # After Concurso 1
    assert result_df['Atraso_1'].iloc[0] == 0
    assert result_df['Atraso_7'].iloc[0] == 1
    assert result_df['Atraso_60'].iloc[0] == 1
    
    # After Concurso 2
    assert result_df['Atraso_1'].iloc[1] == 1
    assert result_df['Atraso_7'].iloc[1] == 0
    assert result_df['Atraso_60'].iloc[1] == 2

@patch('src.etl.requests.get')
@patch('src.etl.pd.read_excel')
def test_download_data(mock_read_excel, mock_get):
    # Setup mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'fake_excel_content'
    mock_get.return_value = mock_response
    
    # Setup mock read_excel
    mock_df = pd.DataFrame({'Concurso': [1]})
    mock_read_excel.return_value = mock_df
    
    url = "http://fake-url.com"
    df = download_data(url)
    
    mock_get.assert_called_once_with(url)
    mock_read_excel.assert_called_once()
    assert df.equals(mock_df)





