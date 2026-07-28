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

## Baixar banco de dados

Acesse o link abaixo e baixe os bancos de dados:

[BANCO DE DADOS](https://drive.google.com/drive/folders/1T6FGM-OGl5sduLNYWg1Z8BuIpXNiNL8E?usp=sharing)

Apos descompactar adicione os bancos na pasta `banco` e siga para os proximos passos.

## Como usar

Para rodar o codigo para a analise baixe as dependencias que estão listadas em `requirements.txt`, após ao baixar as dependencias descompacte o banco de dados disponivel em `banco`. Para baixar as depedencias e rodar o projeto utilize:

```bash
pip install -r requirements.txt
cd codigos
python detectorAnomalia.py
```

Dentro da pasta codigos estará o detector de anomalias, nele você pode testar varios banco de dados em que sua base seja numerica.
Para atualizar o banco de analise modifique nesta parte:

linha 16
```bash
DATA_PATH = "../banco/SEU_BANCO_AQUI"
```

A taxa de `CONTAMINATION` recomendada para o detecção de anomalia mais precisa é de 1% a 5% (0.01 a 0.05)

linha 18
```bash
CONTAMINATION = 0.02
```

Para bancos de dados massivos com valor grandes é mais recomendado utilizar `RobustScaler` para ter uma analise de dados mais preciso e realista, em bancos mais simples é recomendado utilizar `StandardScaler` para a analise.

linha 57
```bash
X = RobustScaler().fit_transform(df[feature_cols])
```

Saida esta configurada para ao terminar a analise enviar um ".png" com um gráfico e uma planilha com os resultados.
Você podera acessa-los na pasta resultados, caso queira configurar outro local de saida modifique aqui em `detectorAnomalia.py`:

linha 64
```bash
def plot_anomalies(X, is_anomaly, path="../resultados/anomalias.png"):
```

linha 90
```bash
df.to_csv("../resultados/resultado_anomalias.csv", index=False)
```

### Configuração

No topo do arquivo `detectorAnomalia.py`:

- `LABEL_COL` — nome da coluna de rótulo, se o dataset tiver uma (ex: `"Class"`). Deixe `None` se não houver.
- `CONTAMINATION` — proporção esperada de anomalias no dataset.

## Saída gerada

- `anomalias.png` — gráfico de dispersão com os registros normais e anômalos
- `resultado_anomalias.csv` — dataset original com as colunas `anomaly_score` e `is_anomaly` adicionadas

## Dependências

Ver `requirements.txt`. Principais bibliotecas: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

