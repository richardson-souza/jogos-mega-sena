### Abordagem 1: Algoritmos Genéticos (Otimização Combinatória)

**Metodologia:** Utilizar a teoria da evolução para criar "gerações" de apostas. As apostas (cromossomos) cruzam entre si e sofrem mutações. A "função de aptidão" (fitness) avalia quão bom é um jogo com base em regras heurísticas.
**Justificativa:** Algoritmos Genéticos são perfeitos para espaços de busca massivos (50 milhões de combinações). Podemos treinar o algoritmo não para "adivinhar" o sorteio, mas para gerar um conjunto de 10 a 20 jogos que possuam a melhor dispersão e cobertura estatística possível.
**Exemplo:** O algoritmo descarta jogos que tenham mais de 4 números na mesma dezena (ex: 21, 22, 25, 28) pois historicamente são raros, favorecendo jogos com números bem distribuídos.

**Instruções de Implementação para o Agente:**

1. **Inicialização:** Gere uma população inicial de 1000 jogos aleatórios (indivíduos), cada um com 6 dezenas.
2. **Função de Fitness:** Crie um *score* para cada jogo considerando:
* Replicar a distribuição histórica Par/Ímpar (idealmente 3/3 ou 4/2).
* Soma das dezenas entre 120 e 210 (onde se concentram 80% dos resultados históricos).
* Penalização pesada se o jogo já saiu em algum concurso passado do `data/processed/mega_sena_features.csv`.


3. **Crossover e Mutação:** Selecione os 20% melhores jogos. Faça o cruzamento (misture os números dos jogos vencedores) e aplique uma taxa de mutação de 5% (trocar um número aleatório).
4. **Convergência:** Repita por 500 gerações.
5. **Output:** Extraia os 10 melhores indivíduos da última geração.
