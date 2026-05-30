import os
import pandas as pd
from src.etl import (
    load_and_clean_data, 
    feature_sum, 
    feature_even_odd, 
    feature_spread, 
    feature_delay_matrix,
    feature_consecutive_sequences,
    feature_std_dev,
    feature_zone_density
)

def main():
    raw_file = "data/raw/Mega-Sena.xlsx"
    output_dir = "data/processed"
    output_file = os.path.join(output_dir, "mega_sena_features.csv")
    output_file_sat = os.path.join(output_dir, "mega_sena_features_saturday.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Iniciando o processo de ETL da Mega-Sena...")
    
    if not os.path.exists(raw_file):
        print(f"Erro crítico: O arquivo de origem local '{raw_file}' não foi encontrado!")
        print("Certifique-se de que ele esteja na pasta especificada.")
        return
        
    try:
        print(f"Lendo dados de referência do arquivo local: {raw_file}")
        df = pd.read_excel(raw_file)
        print("Leitura do arquivo concluída com sucesso.")
    except Exception as e:
        print(f"Erro crítico ao ler o arquivo '{raw_file}': {e}")
        return
            
    print("Iniciando limpeza e formatação dos dados...")
    df_clean = load_and_clean_data(df)
    
    # --- 1. PROCESSO PARA TODOS OS JOGOS ---
    print("\n[Base Completa] Processando todos os jogos...")
    df_full = df_clean.copy()
    df_full = feature_sum(df_full)
    df_full = feature_even_odd(df_full)
    df_full = feature_spread(df_full)
    print("Calculando Matriz de Atraso para a base completa...")
    df_full = feature_delay_matrix(df_full)
    df_full = feature_consecutive_sequences(df_full)
    df_full = feature_std_dev(df_full)
    df_full = feature_zone_density(df_full)
    
    print(f"Salvando dataset final da base completa em: {output_file}")
    df_full.to_csv(output_file, index=False)
    print(f"Dimensões do dataset completo: {df_full.shape}")
    
    # --- 2. PROCESSO PARA JOGOS DE SÁBADO ---
    print("\n[Base de Sábado] Filtrando e processando apenas jogos realizados aos sábados...")
    # Converter para datetime temporariamente para filtrar
    df_clean['Data_dt'] = pd.to_datetime(df_clean['Data do Sorteio'], format='%d/%m/%Y')
    df_sat = df_clean[df_clean['Data_dt'].dt.weekday == 5].copy()
    df_sat = df_sat.drop(columns=['Data_dt']).reset_index(drop=True)
    
    if len(df_sat) == 0:
        print("Aviso: Nenhum jogo realizado no sábado foi encontrado!")
    else:
        df_sat = feature_sum(df_sat)
        df_sat = feature_even_odd(df_sat)
        df_sat = feature_spread(df_sat)
        print("Calculando Matriz de Atraso para os sábados...")
        df_sat = feature_delay_matrix(df_sat)
        df_sat = feature_consecutive_sequences(df_sat)
        df_sat = feature_std_dev(df_sat)
        df_sat = feature_zone_density(df_sat)
        
        print(f"Salvando dataset final de sábados em: {output_file_sat}")
        df_sat.to_csv(output_file_sat, index=False)
        print(f"Dimensões do dataset de sábados: {df_sat.shape}")
        
    print("\nProcesso finalizado com sucesso!")

if __name__ == "__main__":
    main()
