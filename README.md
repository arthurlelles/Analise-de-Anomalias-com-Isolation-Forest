# Detecção de Anomalias com Isolation Forest

Projeto final da disciplina de Banco de Dados II (UFU) — tema: Aprendizado de Máquina Não Supervisionado.

## Objetivo

Este projeto implementa um script genérico de **detecção de anomalias** em dados tabulares, utilizando o algoritmo **Isolation Forest**. Diferente de algoritmos supervisionados, o modelo não recebe nenhum exemplo rotulado durante o treinamento — ele identifica registros anômalos apenas observando a estrutura dos dados, o que o torna aplicável a qualquer conjunto de dados numérico, de qualquer domínio (fraude financeira, sensores, logs de sistema, etc.), sem alterações no código.

## Como funciona

1. **Carregamento dos dados** — lê um arquivo CSV informado pelo usuário. Se nenhum arquivo for encontrado, gera um conjunto de dados sintético para fins de teste.
2. **Seleção de atributos** — identifica automaticamente todas as colunas numéricas como entrada, ignorando a coluna de rótulo (se houver uma).
3. **Padronização** — normaliza os atributos com `StandardScaler`, colocando todas as variáveis na mesma escala.
4. **Treinamento** — treina o modelo `IsolationForest` (scikit-learn) sobre os dados padronizados, sem usar rótulos.
5. **Classificação** — cada registro recebe um score de anomalia e é classificado como normal ou anômalo.
6. **Visualização** — reduz os atributos para 2 dimensões via PCA (quando necessário) e gera um gráfico de dispersão destacando os pontos anômalos.
7. **Avaliação (opcional)** — se o dataset tiver uma coluna de rótulo real, calcula métricas de precisão e revocação, usadas apenas para validar o resultado, nunca durante o treinamento.
8. **Exportação** — salva um novo CSV com os resultados (score e classificação de cada registro).

## Como usar

```bash
pip install -r requirements.txt
python deteccao_anomalias_generico.py caminho/para/seus_dados.csv
```

Se nenhum caminho for informado, o script roda com dados sintéticos gerados automaticamente:

```bash
python deteccao_anomalias_generico.py
```

### Configuração

No topo do arquivo `deteccao_anomalias_generico.py`:

- `LABEL_COL` — nome da coluna de rótulo, se o dataset tiver uma (ex: `"Class"`). Deixe `None` se não houver.
- `CONTAMINATION` — proporção esperada de anomalias no dataset. Use `"auto"` para deixar o scikit-learn estimar, ou um valor entre 0 e 0.5 (ex: `0.02` para 2%).

## Saída gerada

- `anomalias.png` — gráfico de dispersão com os registros normais e anômalos
- `resultado_anomalias.csv` — dataset original com as colunas `anomaly_score` e `is_anomaly` adicionadas

## Dependências

Ver `requirements.txt`. Principais bibliotecas: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

## Dataset recomendado para testes

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — transações de cartão de crédito com rótulo de fraude confirmada, usado como referência principal deste trabalho.
