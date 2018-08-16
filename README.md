# Propósito
Notebook com experimentos utilizados no mestrado.

## Notebooks utilizados

| Notebook      | Descrição     |
| ------------- | ------------- |
| Exame dos dados de anomalia SST      | Utilizado para entender geração de anomalia, e gráfico de anomalia cumulativa para verificação de gráfico de tendência. |
| Rainfall forecast<REGION>     | Notebook com teste de algoritmos de ML para previsão de chuva nesta região     |
| Rainfall climatology, anomaly and cumulative | Aprendendo conceitos sobre climatologia e anomalia      |
| Lendo dados pluviométricos da FUNCEME | Apenas estudos      |
| Rainfall climatology, anomaly and cumulative | Apenas estudos      |
| Lendo dados de anomalia da FUNCEME | Apenas estudos      |


# Experimentos de datascience com dados do PIRATA/FUNCEME.

Datasets foram obtidos do site da [FUNCEME](http://www.funceme.br/index.php/areas/19-monitoramento/oceanogr%C3%A1fico/403-campos-de-tsm-e-vento-no-atlantico-tropical).

No notebook "Nova anomalia" os dados so carregados no [formato da FUNCEME](http://www.funceme.br/index.php/areas/19-monitoramento/oceanogr%C3%A1fico/403-campos-de-tsm-e-vento-no-atlantico-tropical). Inicialmente é lido o arquivo maior com dados de janeiro de 1964 até abril de 2014. 
Após isso é feito a adiço dos arquivos individuais de cada mês até dezembro de 2017.

Há um tratamento nos dados: adição de índices por data, e uso dos dados de janeiro de 2009 até dezembro de 2017.

Da é feito cálculo de climatologia, média e anomalia.
