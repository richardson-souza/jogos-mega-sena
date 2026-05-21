### Abordagem 3: Regras de Associação e Clusterização (Aprendizado Não Supervisionado)

**Metodologia:** Combinar o algoritmo Apriori (famoso por análise de carrinhos de compra) para achar dezenas que costumam sair juntas, e *K-Means* para agrupar o comportamento dos sorteios.
**Justificativa:** É a melhor forma de explorar a "simpatia" entre números. Algumas dezenas podem sair juntas com mais frequência devido a características puramente caóticas. Além disso, garante que os jogos criados representem fielmente o comportamento global do sistema.
**Exemplo:** O modelo descobre a regra "Se 14 e 23 saem, 51 tem alta chance de sair" (Itemsets frequentes).

**Instruções de Implementação para o Agente:**

1. **Regras de Associação:** Aplique o algoritmo Apriori ou FP-Growth nas colunas `Bola1` a `Bola6`. Extraia regras com um `Support` e `Confidence` mínimos que representem pares ou trincas de números frequentes.
2. **Clusterização:** Use *K-Means* no dataset (baseado na soma, desvio padrão das bolas de um mesmo jogo, etc.) para achar `K=5` perfis de sorteios da Mega-Sena.
3. **Geração Sintética:** Para cada um dos 5 perfis (clusters), crie 2 jogos âncoras. Preencha esses jogos primeiramente usando as trincas extraídas do Apriori e complete com números selecionados aleatoriamente dentro da distribuição estatística daquele cluster.
4. **Output:** Um *pool* diversificado de jogos que cobre múltiplos comportamentos possíveis do globo de sorteio.