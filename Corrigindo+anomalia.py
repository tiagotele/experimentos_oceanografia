# coding: utf-8

# # Infraestrutura inicial(imports, funções auxiliares, etc)

# In[1]:


import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from statsmodels.tsa import seasonal
from matplotlib import pyplot as plt
from functools import reduce  # Operação de reduce para cálculo de média de uma lista
from datetime import datetime

# get_ipython().magic('matplotlib inline')

# Configurações para exibição de tableas
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Nomes das colunas adicionadas ao dataframe
COLUNA_ANOMALIA_ACUMULADA = "anomalia_acumulada"
COLUNA_ANOMALIA_DO_MES = "anomalia_mensal"
COLUNA_MEDIA_MENSAL = "media_mensal"

# Todo mês possui 25 linhas por 38 colunas que dá 950
BLOCO_DE_DADOS_DE_UM_MES = 950
QUANTIDADE_DE_VALORES_DO_ARQUIVO = 573800  # (950 blocos x 604 meses,01/1964 até 04/2014)


def constroi_colunas_latitude_longitude():
    w_values = range(60, 0, -2)
    e_values = range(0, 15, 2)
    n_values = range(30, 0, -2)
    s_values = range(0, 20, 2)

    colunas = []
    linhas = []

    # LINHAS
    for value in w_values:
        column_name = str(value) + "W"
        linhas.append(column_name)

    for value in e_values:
        column_name = str(value) + "E"
        linhas.append(column_name)

    # COLUNAS
    for value in n_values:
        column_name = str(value) + "N"
        colunas.append(column_name)

    for value in s_values:
        column_name = str(value) + "S"
        colunas.append(column_name)

    colunas_do_data_frame = []
    for linha in linhas:
        for coluna in colunas:
            lat_long = linha + "-" + coluna
            colunas_do_data_frame.append(lat_long)
    return colunas_do_data_frame


def carrega_array_com_valores_do_arquivo_geral(
        arquivo_com_decadas_de_anomalia="funceme_db/tsm/geral/_Dados_TSMvento_2014_04_sst6414b04"):
    global QUANTIDADE_DE_VALORES_DO_ARQUIVO

    conteudo_do_arquivo = open(arquivo_com_decadas_de_anomalia).read()
    conteudo_do_arquivo = conteudo_do_arquivo.replace("\n", "")

    # Carrega todos os dados de anomalia em um único array
    qtd_char_no_arquivo = 5
    # Todos os valores do arquivo em um único array. Não há separação de mês. Tudo está de forma sequencial
    valores_do_arquivo = []
    for rows_index in range(QUANTIDADE_DE_VALORES_DO_ARQUIVO):
        # slice data like (n:n+5)
        value = float(conteudo_do_arquivo[
                      rows_index * qtd_char_no_arquivo: rows_index * qtd_char_no_arquivo + qtd_char_no_arquivo].replace(
            " ", ""))
        value /= 10
        valores_do_arquivo.append(value)
    return valores_do_arquivo


def carrega_array_com_valores_do_arquivo_mensal(file_name):
    file_content = open(file_name).read()

    # Remove header de um único arquivo
    file_content = file_content[25:]
    file_content = file_content.replace("\n", "")

    block_size = 5
    dados_do_arquivo = []
    print("arquivo ", file_name)
    for rows_index in range(BLOCO_DE_DADOS_DE_UM_MES):
        # slice data like (n:n+5)
        content = file_content[rows_index * block_size: rows_index * block_size + block_size].replace(" ", "")
        print(" content ", content)
        value = float(content)
        value /= 10
        dados_do_arquivo.append(value)

    return dados_do_arquivo


def merge_dados_do_diretorio(diretorio_anomalia_individual="funceme_db/tsm/individual/"):
    global QUANTIDADE_DE_VALORES_DO_ARQUIVO
    lista_de_arquivos_individuais = []
    arquivos_do_diretorio = os.listdir(diretorio_anomalia_individual)

    quantidade_de_arquivos = 0
    # Adiciona apenas arquivos com extensão .22
    for arquivo in arquivos_do_diretorio:
        if arquivo.endswith(".22"):
            lista_de_arquivos_individuais.append(arquivo)
            quantidade_de_arquivos += 1

    valores_dos_arquivos = carrega_array_com_valores_do_arquivo_geral()
    # Para cada arquivo na lista é feito append na lista full_data
    for arquivo in lista_de_arquivos_individuais:
        dados_mensais = carrega_array_com_valores_do_arquivo_mensal(diretorio_anomalia_individual + arquivo)
        for item in dados_mensais:
            valores_dos_arquivos.append(item)

    ##44 meses de 05/2014 até 12/2017
    QUANTIDADE_DE_VALORES_DO_ARQUIVO += quantidade_de_arquivos * BLOCO_DE_DADOS_DE_UM_MES  # Que dá 41800

    array_de_anomalias_por_mes = []
    for i in range(0, QUANTIDADE_DE_VALORES_DO_ARQUIVO, 950):
        anomalias_do_mes = valores_dos_arquivos[i:i + BLOCO_DE_DADOS_DE_UM_MES]
        array_de_anomalias_por_mes.append(anomalias_do_mes)
    return array_de_anomalias_por_mes


def inicia_funceme_data_frame():
    funceme_df = pd.DataFrame()
    for anomalias_do_mes in array_de_anomalias_por_mes:
        data = np.array(anomalias_do_mes)
        row_df = pd.DataFrame(data.reshape(-1, len(data)), columns=constroi_colunas_latitude_longitude())
        funceme_df = funceme_df.append(row_df)
    funceme_df.index = range(0, len(array_de_anomalias_por_mes), 1)
    # ### Setando indices baseados na data
    FORMAT = "%Y-%m"
    some_date_time1 = "1964-01"
    data_inicial = datetime.strptime(some_date_time1, FORMAT)
    indexes_data = []
    for i in range(len(array_de_anomalias_por_mes)):
        indexes_data.append(data_inicial + relativedelta(months=i))
    funceme_df = funceme_df.set_index(pd.DatetimeIndex(data=indexes_data))

    return funceme_df


# # Carregando dados da FUNCEME

# In[2]:


array_de_anomalias_por_mes = merge_dados_do_diretorio()
funceme_df = inicia_funceme_data_frame()

# In[ ]:


funceme_df.head(5)

# ## Filtra dados por período jan/2009 até dez/2017

# In[ ]:


funceme_df = funceme_df.loc['2009-01-01':'2017-12-01']
funceme_df.head(5)

# ## Substitui área terrestre por valor não numérico(NAN)

# In[ ]:


funceme_df = funceme_df.replace(9999.8, np.nan)
funceme_df.head(5)

# ## Remove área terrestre(99998)

# In[ ]:


funceme_df = funceme_df.dropna(axis=1, how='any')
funceme_df.head(5)

# ## Adiciona coluna com média mensal

# In[ ]:


# Cria média para cada mês adicionando a coluna "media_mensal"
media_da_figura_no_mes = []

for date, row in funceme_df.iterrows():
    media = (reduce(lambda x, y: x + y, row) / len(row))
    media_da_figura_no_mes.append(media)

# Cria nova coluna chamada media_mensal e adiciona ao dataframe da funceme(funceme_df)
funceme_df.loc[:, "%s" % COLUNA_MEDIA_MENSAL] = pd.Series(media_da_figura_no_mes, index=funceme_df.index)
funceme_df.head(2)

# ## Cálculo de climatologia

# In[ ]:


# Cria dicionário de médias anuais
medias_anuais = {}
for mes in range(1, 13, 1):
    medias_anuais[mes] = []

for date, row in funceme_df.iterrows():
    datetime = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
    medias_anuais[datetime.month].append(row[('%s' % COLUNA_MEDIA_MENSAL)])

# Calcula climatologia para cada mês
climatologias_mensais = {}
for mes in range(1, 13, 1):
    climatologias_mensais[mes] = reduce(lambda x, y: x + y, medias_anuais[mes]) / len(medias_anuais[mes])
climatologias_mensais

# ## Calculando anomalias

# In[ ]:


### Calculando anomalias
anomalia = []
for date, row in funceme_df.iterrows():
    datetime = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
    anomalia_do_mes = row[COLUNA_MEDIA_MENSAL] - climatologias_mensais[datetime.month]
    anomalia.append(anomalia_do_mes)
funceme_df.loc[:, COLUNA_ANOMALIA_DO_MES] = pd.Series(anomalia, index=funceme_df.index)
funceme_df.head(2)

# ## Calculando anomalia acumulada

# In[ ]:


anomalia_acumulada = []
# Calcula anomalia acumulada
for index in range(len(funceme_df.index)):
    if index == 0:
        anomalia_acumulada.append(funceme_df.iloc[index][COLUNA_ANOMALIA_DO_MES])
        continue
    anomalia_acumulada.append(
        funceme_df.iloc[index][COLUNA_ANOMALIA_DO_MES] + funceme_df.iloc[index - 1][COLUNA_ANOMALIA_DO_MES])

funceme_df.loc[:, COLUNA_ANOMALIA_ACUMULADA] = pd.Series(anomalia_acumulada, index=funceme_df.index)
funceme_df.head(5)

# ## Plota gráficos

# In[ ]:


fig, axarr = plt.subplots(2)
fig.set_size_inches(10, 10)

funceme_df[COLUNA_ANOMALIA_DO_MES].plot(ax=axarr[0], color='b', linestyle='-')
axarr[0].set_title('Anomalia mensal')

funceme_df[COLUNA_ANOMALIA_ACUMULADA].plot(color='r', linestyle='-', ax=axarr[1])
axarr[1].set_title('Anomalia acumulada')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
plt.xticks(rotation=10)
plt.show()

# In[ ]:


funceme_df.loc[:, funceme_df.columns.str.contains('_')]
