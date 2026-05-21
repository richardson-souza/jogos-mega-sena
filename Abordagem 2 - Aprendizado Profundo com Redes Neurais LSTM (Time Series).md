### Abordagem 2: Aprendizado Profundo com Redes Neurais LSTM (Time Series)

**Metodologia:** Tratar os sorteios como uma sequência temporal multivariada. Usar uma rede Long Short-Term Memory (LSTM) para prever as "probabilidades" ou o "calor" de cada número (1 a 60) no próximo instante de tempo `t+1`.
**Justificativa:** LSTMs são excelentes em encontrar correlações temporais não-lineares ocultas. Se houver qualquer desgaste físico nas bolas, viés na máquina de sorteio, ou autocorrelação matemática sutil, a LSTM vai detectar a flutuação das frequências.
**Exemplo:** A rede pode identificar que a "Bola 42" entra em um período quente após estar ausente por 15 sorteios, quando pareada com números na casa dos 20.

**Instruções de Implementação para o Agente:**

1. **Transformação do Dataset:** Converta o `sample.csv` em um formato *One-Hot Encoded* ou de frequências cumulativas, onde cada linha é um vetor de tamanho 60. O valor 1 indica se a bola saiu, 0 se não.
2. **Janela Deslizante:** Crie sequências temporais de janela fixa (ex: os últimos 20 sorteios preveem os números do 21º sorteio).
3. **Arquitetura da Rede:** * Camada de Entrada: `(Tamanho da Janela, 60)`.
* Camadas LSTM Ocultas com Dropout para evitar *overfitting*.
* Camada de Saída: `Dense` com 60 neurônios e ativação `Sigmoid` (retornando uma probabilidade para cada número de 1 a 60).


4. **Geração:** Após o treinamento, passe os últimos 20 sorteios reais do dataset. Pegue as 15 dezenas com maior probabilidade de saída.
5. **Output:** Faça combinações desdobradas (matriz combinatória) dessas 15 dezenas, resultando em um conjunto otimizado de cartelas.