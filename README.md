Experimentos de datascience com dados do PIRATA/FUNCEME.

Datasets foram obtidos do site da [FUNCEME](http://www.funceme.br/index.php/areas/19-monitoramento/oceanogr%C3%A1fico/403-campos-de-tsm-e-vento-no-atlantico-tropical).

No notebook "Nova anomalia" os dados so carregados no [formato da FUNCEME](http://www.funceme.br/index.php/areas/19-monitoramento/oceanogr%C3%A1fico/403-campos-de-tsm-e-vento-no-atlantico-tropical). Inicialmente é lido o arquivo maior com dados de janeiro de 1964 até abril de 2014. 
Após isso é feito a adiço dos arquivos individuais de cada mês até dezembro de 2017.

Há um tratamento nos dados: adição de índices por data, e uso dos dados de janeiro de 2009 até dezembro de 2017.

Da é feito cálculo de climatologia, média e anomalia.
