import os
import pandas as pd
from src.etl import (
    download_data, 
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
    url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
    output_dir = "data/processed"
    output_file = os.path.join(output_dir, "mega_sena_features.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Iniciando o processo de ETL da Mega-Sena...")
    
    try:
        print(f"Baixando dados de: {url}")
        df = download_data(url)
        print("Download concluído com sucesso.")
    except Exception as e:
        print(f"Erro no download: {e}")
        print("Tentando usar 'sample.csv' localmente como fallback...")
        if os.path.exists("sample.csv"):
            df = pd.read_csv("sample.csv")
        else:
            print("Arquivo sample.csv não encontrado. Encerrando.")
            return
            
    print("Iniciando limpeza e formatação dos dados...")
    df = load_and_clean_data(df)
    
    print("Calculando Soma...")
    df = feature_sum(df)
    
    print("Calculando Proporção de Pares e Ímpares...")
    df = feature_even_odd(df)
    
    print("Calculando Spread (Distância)...")
    df = feature_spread(df)
    
    print("Calculando Matriz de Atraso (Isso pode demorar alguns segundos)...")
    df = feature_delay_matrix(df)
    
    print("Calculando Consecutividade...")
    df = feature_consecutive_sequences(df)
    
    print("Calculando Desvio Padrão...")
    df = feature_std_dev(df)
    
    print("Calculando Densidade de Zona...")
    df = feature_zone_density(df)
    
    print(f"Salvando dataset final com as features em: {output_file}")
    df.to_csv(output_file, index=False)
    
    print("Processo finalizado com sucesso!")
    print(f"Dimensões do dataset gerado: {df.shape}")

if __name__ == "__main__":
    main()
