import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from statsmodels.tsa import seasonal
from matplotlib import pyplot as plt
from functools import reduce  # Operação de reduce para cálculo de média de uma lista
from datetime import datetime

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


def carrega_array_com_valores_do_arquivo_geral(
        arquivo_com_decadas_de_anomalia="funceme_db/anomalia/geral/_Dados_TSMvento_2014_04_anomt6414b04"):
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
                      rows_index * qtd_char_no_arquivo: rows_index * qtd_char_no_arquivo + qtd_char_no_arquivo])
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
    for rows_index in range(BLOCO_DE_DADOS_DE_UM_MES):
        # slice data like (n:n+5)
        value = float(file_content[rows_index * block_size: rows_index * block_size + block_size])
        value /= 10
        dados_do_arquivo.append(value)

    return dados_do_arquivo


def merge_dados_do_diretorio(diretorio_anomalia_individual="funceme_db/anomalia/individual/"):
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

    print("Quantidade de arquivos ", quantidade_de_arquivos)

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
    funceme_df = funceme_df.loc['2009-01-01':'2017-12-01']

    funceme_df = funceme_df.replace(9999.8, np.nan)

    funceme_df.dropna(axis=1, how='all', inplace=True)
    return funceme_df


def monta_grafico_de_decomposicao(coluna):
    decompose_model = seasonal.seasonal_decompose(funceme_df[coluna].tolist(), freq=12,
                                                  model='additive')
    fig, axarr = plt.subplots(4, sharex=True)
    fig.set_size_inches(5.5, 5.5)

    funceme_df[coluna].plot(ax=axarr[0], color='b', linestyle='-')
    axarr[0].set_title('Gráfico mensal')

    pd.Series(data=decompose_model.trend, index=funceme_df.index).plot(color='r', linestyle='-', ax=axarr[1])
    axarr[1].set_title('Gráfico de Tendência')

    pd.Series(data=decompose_model.seasonal, index=funceme_df.index).plot(color='g', linestyle='-', ax=axarr[2])
    axarr[2].set_title('Componente Sazonal')

    pd.Series(data=decompose_model.resid, index=funceme_df.index).plot(color='k', linestyle='-', ax=axarr[3])
    axarr[3].set_title('Variações irregulares')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    plt.xticks(rotation=10)


if __name__ == '__main__':
    array_de_anomalias_por_mes = merge_dados_do_diretorio()
    funceme_df = inicia_funceme_data_frame()

    ####################################### ATIVIDADE DO GERALDO ######################################

    # Cria média para cada mês adicionando a coluna "media_mensal"
    medias_mensais = []

    for date, row in funceme_df.iterrows():
        media = (reduce(lambda x, y: x + y, row) / len(row))
        medias_mensais.append(media)

    #Cria nova coluna chamada media_mensal e adiciona ao dataframe da funceme(funceme_df)
    funceme_df.loc[:, "media_mensal"] = pd.Series(medias_mensais, index=funceme_df.index)

    # Salva médias em array de médias
    medias_anuais_janeiro = []
    medias_anuais_fevereiro = []
    medias_anuais_marco = []
    medias_anuais_abril = []
    medias_anuais_maio = []
    medias_anuais_junho = []
    medias_anuais_julho = []
    medias_anuais_agosto = []
    medias_anuais_setembro = []
    medias_anuais_outubro = []
    medias_anuais_novembro = []
    medias_anuais_dezembro = []

    for date, row in funceme_df.iterrows():
        datetime_object = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
        if datetime_object.month == 1:
            medias_anuais_janeiro.append(row['media_mensal'])
        if datetime_object.month == 2:
            medias_anuais_fevereiro.append(row['media_mensal'])
        if datetime_object.month == 3:
            medias_anuais_marco.append(row['media_mensal'])
        if datetime_object.month == 4:
            medias_anuais_abril.append(row['media_mensal'])
        if datetime_object.month == 5:
            medias_anuais_maio.append(row['media_mensal'])
        if datetime_object.month == 6:
            medias_anuais_junho.append(row['media_mensal'])
        if datetime_object.month == 7:
            medias_anuais_julho.append(row['media_mensal'])
        if datetime_object.month == 8:
            medias_anuais_agosto.append(row['media_mensal'])
        if datetime_object.month == 9:
            medias_anuais_setembro.append(row['media_mensal'])
        if datetime_object.month == 10:
            medias_anuais_outubro.append(row['media_mensal'])
        if datetime_object.month == 11:
            medias_anuais_novembro.append(row['media_mensal'])
        if datetime_object.month == 12:
            medias_anuais_dezembro.append(row['media_mensal'])

    # Calcula climatologia para cada mês

    climatologia_janeiro = (reduce(lambda x, y: x + y, medias_anuais_janeiro) / len(medias_anuais_janeiro))
    climatologia_fevereiro = (reduce(lambda x, y: x + y, medias_anuais_fevereiro) / len(medias_anuais_fevereiro))
    climatologia_marco = (reduce(lambda x, y: x + y, medias_anuais_marco) / len(medias_anuais_marco))
    climatologia_abril = (reduce(lambda x, y: x + y, medias_anuais_abril) / len(medias_anuais_abril))
    climatologia_maio = (reduce(lambda x, y: x + y, medias_anuais_maio) / len(medias_anuais_maio))
    climatologia_junho = (reduce(lambda x, y: x + y, medias_anuais_junho) / len(medias_anuais_junho))
    climatologia_julho = (reduce(lambda x, y: x + y, medias_anuais_julho) / len(medias_anuais_julho))
    climatologia_agosto = (reduce(lambda x, y: x + y, medias_anuais_agosto) / len(medias_anuais_agosto))
    climatologia_setembro = (reduce(lambda x, y: x + y, medias_anuais_setembro) / len(medias_anuais_setembro))
    climatologia_outubro = (reduce(lambda x, y: x + y, medias_anuais_outubro) / len(medias_anuais_outubro))
    climatologia_novembro = (reduce(lambda x, y: x + y, medias_anuais_novembro) / len(medias_anuais_novembro))
    climatologia_dezembro = (reduce(lambda x, y: x + y, medias_anuais_dezembro) / len(medias_anuais_dezembro))

    print("Climatologia de cada mês")
    print(climatologia_janeiro, climatologia_fevereiro, climatologia_marco, climatologia_abril, climatologia_maio,
          climatologia_junho, climatologia_julho, climatologia_agosto, climatologia_setembro, climatologia_outubro,
          climatologia_novembro, climatologia_dezembro)

    ### Calculando anomalias

    anomalia = []
    for date, row in funceme_df.iterrows():
        datetime_object = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
        anomalia_do_mes = 0
        if datetime_object.month == 1:
            anomalia_do_mes = row["media_mensal"] - climatologia_janeiro
        if datetime_object.month == 2:
            anomalia_do_mes = row["media_mensal"] - climatologia_fevereiro
        if datetime_object.month == 3:
            anomalia_do_mes = row["media_mensal"] - climatologia_marco
        if datetime_object.month == 4:
            anomalia_do_mes = row["media_mensal"] - climatologia_abril
        if datetime_object.month == 5:
            anomalia_do_mes = row["media_mensal"] - climatologia_maio
        if datetime_object.month == 6:
            anomalia_do_mes = row["media_mensal"] - climatologia_junho
        if datetime_object.month == 7:
            anomalia_do_mes = row["media_mensal"] - climatologia_julho
        if datetime_object.month == 8:
            anomalia_do_mes = row["media_mensal"] - climatologia_agosto
        if datetime_object.month == 9:
            anomalia_do_mes = row["media_mensal"] - climatologia_setembro
        if datetime_object.month == 10:
            anomalia_do_mes = row["media_mensal"] - climatologia_outubro
        if datetime_object.month == 11:
            anomalia_do_mes = row["media_mensal"] - climatologia_novembro
        if datetime_object.month == 12:
            anomalia_do_mes = row["media_mensal"] - climatologia_dezembro
        anomalia.append(anomalia_do_mes)
    funceme_df.loc[:, "anomalia_mensal"] = pd.Series(anomalia, index=funceme_df.index)

    x = funceme_df.index

    plt.figure(figsize=(12, 6))
    plt.plot(x, funceme_df['anomalia_mensal'], label="anomalia")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.74))
    plt.xticks(rotation=45)
    plt.title("Gráfico")
    plt.show()

    anomalia_acumulada = []
    # Calcula anomalia acumulada
    for index in range(len(funceme_df.index)):
        print(index, funceme_df.iloc[index]["anomalia_mensal"])
        if index == 0:
            anomalia_acumulada.append(funceme_df.iloc[index]["anomalia_mensal"])
            continue
        anomalia_acumulada.append(
            funceme_df.iloc[index]["anomalia_mensal"] + funceme_df.iloc[index - 1]["anomalia_mensal"])

    funceme_df.loc[:, "anomalia_acumulada"] = pd.Series(anomalia_acumulada, index=funceme_df.index)

    print(funceme_df.head(3))

    plt.figure(figsize=(12, 6))
    plt.plot(x, funceme_df['anomalia_acumulada'], label="anomalia acumulada")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.74))
    plt.xticks(rotation=45)
    plt.title("Gráfico")
    plt.show()
