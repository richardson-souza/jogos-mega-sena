import pandas as pd
import requests
from io import BytesIO

def load_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Concurso', 'Data do Sorteio', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    df = df[cols].copy()
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.sort_values(by='Concurso').reset_index(drop=True)
    return df

def feature_sum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    df['Soma'] = df[cols_bolas].sum(axis=1)
    return df

def feature_even_odd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    # Conta pares
    df['Qtd_Pares'] = df[cols_bolas].apply(lambda row: sum(1 for x in row if x % 2 == 0), axis=1)
    # Conta ímpares
    df['Qtd_Impares'] = df[cols_bolas].apply(lambda row: sum(1 for x in row if x % 2 != 0), axis=1)
    
    # Proporções
    df['Prop_Pares'] = df['Qtd_Pares'] / 6.0
    df['Prop_Impares'] = df['Qtd_Impares'] / 6.0
    
    return df

def feature_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    df['Spread'] = df[cols_bolas].max(axis=1) - df[cols_bolas].min(axis=1)
    return df

def feature_delay_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    delays = {i: 0 for i in range(1, 61)}
    delay_records = []
    
    for idx, row in df.iterrows():
        bolas_sorteadas = set(row[cols_bolas].dropna().astype(int))
        
        current_delays = {}
        for num in range(1, 61):
            if num in bolas_sorteadas:
                delays[num] = 0
            else:
                if idx == 0:
                    delays[num] = 1
                else:
                    delays[num] += 1
            current_delays[f'Atraso_{num}'] = delays[num]
            
        delay_records.append(current_delays)
        
    delay_df = pd.DataFrame(delay_records)
    
    # Resetting index to concatenate safely
    df = df.reset_index(drop=True)
    df = pd.concat([df, delay_df], axis=1)
    return df

def feature_consecutive_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    def count_consecutive(row):
        nums = sorted(row.dropna().astype(int).tolist())
        count = 0
        for i in range(len(nums) - 1):
            if nums[i+1] - nums[i] == 1:
                count += 1
        return count
        
    df['Qtd_Consecutivos'] = df[cols_bolas].apply(count_consecutive, axis=1)
    return df

def feature_std_dev(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    df['Desvio_Padrao'] = df[cols_bolas].std(axis=1)
    return df

def feature_zone_density(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    
    def max_zone_density(row):
        nums = row.dropna().astype(int).tolist()
        zones = [0]*6
        for n in nums:
            z = (n - 1) // 10
            if 0 <= z <= 5:
                zones[z] += 1
        return max(zones)
        
    df['Max_Densidade_Zona'] = df[cols_bolas].apply(max_zone_density, axis=1)
    return df

def download_data(url: str) -> pd.DataFrame:
    response = requests.get(url)
    if response.status_code == 200:
        try:
            df = pd.read_excel(BytesIO(response.content))
            return df
        except Exception as e:
            raise Exception(f"Error reading the Excel file: {e}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

