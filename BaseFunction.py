# coding: utf-8

# ### Initial imports

import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from functools import reduce  # Operação de reduce para cálculo de média de uma lista
from datetime import datetime
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from scipy import interp

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

second_semester_months = [7, 8, 9, 10, 11, 12]  # From August to December


class LatitudeColumns:

    def __init__(self):
        self.values = range(29, -21, -2)
        self.positive = 'N'
        self.negative = 'S'

    def get_all_columns(self):
        colum_names = []
        for value in self.values:
            if value > 0:
                colum_name = str(value) + self.positive
            else:
                colum_name = str(value) + self.negative
            colum_names.append(colum_name)
        return colum_names

    def get_single_column(self, desired_value, human_readable=True):
        sub_values = []
        for value in self.values:
            if value != desired_value:
                continue
            sub_values.append(value)
        if human_readable:
            return self.parse_human_readable(sub_values)
        return sub_values

    def get_range(self, init, final, human_readable=True):
        final, init = self.switch(final, init)

        sub_values = []
        if init == final:
            return self.get_single_column(init, human_readable)

        for value in self.values:
            if (value > init) or (value < final):
                continue
            sub_values.append(value)

        if human_readable:
            return self.parse_human_readable(sub_values)

        return sub_values

    def switch(self, final, init):
        if final > init:
            aux = init
            init = final
            final = aux
        return final, init

    def parse_human_readable(self, values):
        colum_names = []
        for value in values:
            if value > 0:
                colum_name = str(value) + self.positive
            else:
                colum_name = str(-value) + self.negative
            colum_names.append(colum_name)
        return colum_names


class LongitudeColumns:

    def __init__(self):
        # From map 60W to 15E(-17 because 'force' go to -15)
        self.values = range(59, -17, -2)
        self.positive = 'W'
        self.negative = 'E'

    def get_all_columns(self):
        colum_names = []
        for value in self.values:
            if value > 0:
                colum_name = str(value) + self.positive
            else:
                colum_name = str(value) + self.negative
            colum_names.append(colum_name)
        return colum_names

    def get_single_column(self, desired_value, human_readable=True):
        sub_values = []
        for value in self.values:
            if value != desired_value:
                continue
            sub_values.append(value)
        if human_readable:
            return self.parse_human_readable(sub_values)
        return sub_values

    def get_range(self, init, final, human_readable=True):
        final, init = self.switch(final, init)

        sub_values = []
        if init == final:
            return self.get_single_column(init, human_readable)

        for value in self.values:
            if (value > init) or (value < final):
                continue
            sub_values.append(value)

        if human_readable:
            return self.parse_human_readable(sub_values)

        return sub_values

    def switch(self, final, init):
        if final > init:
            aux = init
            init = final
            final = aux
        return final, init

    def parse_human_readable(self, values):
        colum_names = []
        for value in values:
            if value > 0:
                colum_name = str(value) + self.positive
            else:
                colum_name = str(-value) + self.negative
            colum_names.append(colum_name)
        return colum_names


def constroi_colunas_latitude_longitude(init_lat=29, end_lat=-21,
                                        init_long=59, end_long=-17):
    lat = LatitudeColumns().get_range(init_lat, end_lat)
    long = LongitudeColumns().get_range(init_long, end_long)
    colunas_do_data_frame = []
    for linha in lat:
        for coluna in long:
            lat_long = linha + "-" + coluna
            colunas_do_data_frame.append(lat_long)
    return colunas_do_data_frame


def carrega_array_com_valores_do_arquivo_geral(
        arquivo_com_decadas_de_anomalia="funceme_db/anomalia_tsm/geral/_Dados_TSMvento_2014_04_anomt6414b04"):
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
        value = float("%.3f" % value)
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
        value = float("%.3f" % value)
        value /= 10
        dados_do_arquivo.append(value)

    return dados_do_arquivo


def merge_dados_do_diretorio(diretorio_arquivo_geral, diretorio_arquivo_individual):
    global QUANTIDADE_DE_VALORES_DO_ARQUIVO
    qtde = QUANTIDADE_DE_VALORES_DO_ARQUIVO
    lista_de_arquivos_individuais = []
    arquivos_do_diretorio = os.listdir(diretorio_arquivo_individual)

    quantidade_de_arquivos = 0
    # Adiciona apenas arquivos com extensão .22
    for arquivo in arquivos_do_diretorio:
        if arquivo.endswith(".22"):
            lista_de_arquivos_individuais.append(arquivo)
            quantidade_de_arquivos += 1

    valores_dos_arquivos = carrega_array_com_valores_do_arquivo_geral(diretorio_arquivo_geral)
    # Para cada arquivo na lista é feito append na lista full_data
    for arquivo in lista_de_arquivos_individuais:
        dados_mensais = carrega_array_com_valores_do_arquivo_mensal(diretorio_arquivo_individual + arquivo)
        for item in dados_mensais:
            valores_dos_arquivos.append(item)

    ##44 meses de 05/2014 até 12/2017
    qtde += quantidade_de_arquivos * BLOCO_DE_DADOS_DE_UM_MES  # Que dá 41800

    array_de_anomalias_por_mes = []
    for i in range(0, qtde, 950):
        anomalias_do_mes = valores_dos_arquivos[i:i + BLOCO_DE_DADOS_DE_UM_MES]
        array_de_anomalias_por_mes.append(anomalias_do_mes)
    return array_de_anomalias_por_mes


def inicia_funceme_data_frame(array_de_anomalias_por_mes):
    funceme_rainy_df = pd.DataFrame()
    for anomalias_do_mes in array_de_anomalias_por_mes:
        data = np.array(anomalias_do_mes)
        row_df = pd.DataFrame(data.reshape(-1, len(data)), columns=constroi_colunas_latitude_longitude())
        funceme_rainy_df = funceme_rainy_df.append(row_df)
    funceme_rainy_df.index = range(0, len(array_de_anomalias_por_mes), 1)
    # ### Setando indices baseados na data
    format = "%Y-%m"
    some_date_time1 = "1964-01"
    data_inicial = datetime.strptime(some_date_time1, format)
    indexes_data = []
    for i in range(len(array_de_anomalias_por_mes)):
        indexes_data.append(data_inicial + relativedelta(months=i))
    funceme_rainy_df = funceme_rainy_df.set_index(pd.DatetimeIndex(data=indexes_data))

    return funceme_rainy_df


def plota_coluna_do_dataframe(dataframe, titulo, nome_da_coluna, save_figure=False):
    fig, axarr = plt.subplots(1)
    fig.set_size_inches(8, 5)
    ax = dataframe[nome_da_coluna].plot(color='b', linestyle='-', grid=True)
    ax.set(xlabel="Year", ylabel="Celsius/10")

    plt.title(titulo)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    plt.axhline(0, color='black')
    if save_figure:
        plt.savefig("Imagens/tsm/" + titulo)
    else:
        plt.show()

    plt.close()


def adiciona_media_mensal(dataframe):
    # Cria média para cada mês adicionando a coluna "media_mensal"
    media_da_figura_no_mes = []

    for date, row in dataframe.iterrows():
        media = (reduce(lambda x, y: x + y, row) / len(row))
        media_da_figura_no_mes.append(media)

    # Cria nova coluna chamada media_mensal e adiciona ao dataframe da funceme(funceme_rainy_df)
    dataframe.loc[:, "%s" % COLUNA_MEDIA_MENSAL] = pd.Series(media_da_figura_no_mes, index=dataframe.index)
    return dataframe


def calcula_climatologia_para_dataframe(dataframe):
    # Cria dicionário de médias anuais
    medias_anuais = {}
    for mes in range(1, 13, 1):
        medias_anuais[mes] = []

    for date, row in dataframe.iterrows():
        data = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
        medias_anuais[data.month].append(row[('%s' % COLUNA_MEDIA_MENSAL)])

    # Calcula climatologia para cada mês
    climatologias_mensais = {}
    for mes in range(1, 13, 1):
        climatologias_mensais[mes] = reduce(lambda x, y: x + y, medias_anuais[mes]) / len(medias_anuais[mes])
    return climatologias_mensais


def adiciona_anomalia(dataframe):
    climatologias_mensais = calcula_climatologia_para_dataframe(dataframe)

    anomalia = []
    for date, row in dataframe.iterrows():
        data = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
        anomalia_do_mes = row[COLUNA_MEDIA_MENSAL] - climatologias_mensais[data.month]
        anomalia.append(anomalia_do_mes)
    dataframe.loc[:, COLUNA_ANOMALIA_DO_MES] = pd.Series(anomalia, index=dataframe.index)
    dataframe.head(2)
    return dataframe


def adiciona_anomalia_acumulada(dataframe):
    anomalia_acumulada = []
    # Calcula anomalia acumulada
    for index in range(len(dataframe.index)):
        if index == 0:
            anomalia_acumulada.append(dataframe.iloc[index][COLUNA_ANOMALIA_DO_MES])
            continue
        anterior = anomalia_acumulada[index - 1]
        atual = dataframe.iloc[index][COLUNA_ANOMALIA_DO_MES]
        anomalia_acumulada.append(float("%.3f" % (atual + anterior)))

    dataframe.loc[:, COLUNA_ANOMALIA_ACUMULADA] = pd.Series(anomalia_acumulada, index=dataframe.index)
    return dataframe


def build_rainfall_classified_df(file="scraping/funceme_media_macrorregiao.csv"):
    rainy_seasonal_months = [2, 3, 4]  # February, March, April

    indexes = []
    rows = []
    # Obtém média para cada coluna dos dados pluviométricos
    medias = []
    observado = []
    desvio = []

    funceme_rainy_df = pd.read_csv(file, index_col=0, parse_dates=['datahora'])
    media_observado_a_substituir = funceme_rainy_df.loc['1973-08-01 12:00:00']['Observado(mm)'].mean()
    media_desvio_a_substituir = funceme_rainy_df.loc['1973-08-01 12:00:00']['Desvio(%)'].mean()

    # Trata missing number
    funceme_rainy_df['Normal(mm)'].fillna(media_observado_a_substituir, inplace=True)
    funceme_rainy_df['Observado(mm)'].fillna(media_observado_a_substituir, inplace=True)
    funceme_rainy_df['Desvio(%)'].fillna(media_desvio_a_substituir, inplace=True)

    for indices_unicos in funceme_rainy_df.index.unique():
        medias.append(funceme_rainy_df.loc[indices_unicos]['Normal(mm)'].mean())
        observado.append(funceme_rainy_df.loc[indices_unicos]['Observado(mm)'].mean())
        desvio.append(funceme_rainy_df.loc[indices_unicos]['Desvio(%)'].mean())

        # Cria novo Pandas Dataframe
    funceme_rainy_montlhy_df = pd.DataFrame(index=funceme_rainy_df.index.unique().tolist())

    # Adiciona dados mensais do estado ao Dataframe criado
    funceme_rainy_montlhy_df = pd.concat(
        [funceme_rainy_montlhy_df,
         pd.DataFrame(data=medias, index=funceme_rainy_montlhy_df.index, columns=['Normal(mm)']),
         pd.DataFrame(data=observado, index=funceme_rainy_montlhy_df.index, columns=['Observado(mm)']),
         pd.DataFrame(data=desvio, index=funceme_rainy_montlhy_df.index, columns=['Desvio(%)'])],
        axis=1, join_axes=[funceme_rainy_montlhy_df.index])
    funceme_rainy_montlhy_df.head()

    for index, row in funceme_rainy_montlhy_df.iterrows():
        if index.month not in rainy_seasonal_months:
            continue
        indexes.append(index)
        rows.append(row)

    rainy_classified_df = pd.DataFrame(index=indexes, columns=funceme_rainy_montlhy_df.columns, data=rows)

    rainy_classified_df = rainy_classified_df.groupby(rainy_classified_df.index.year).mean()

    strong = 'strong'
    normal = 'normal'
    weak = 'weak'

    classes = []

    for index, row in rainy_classified_df.iterrows():
        #     print(index)
        if row['Observado(mm)'] < 200:
            classes.append(weak)
            continue

        if row['Observado(mm)'] < 300:
            classes.append(normal)
            continue

        classes.append(strong)
    rainy_classified_df['classes'] = classes
    rainy_classified_df = rainy_classified_df['classes']
    return rainy_classified_df


def create_transformed_df(dataframe, column_name_prefix, initial_year=1973, end_year=2018):
    columns = dataframe.columns
    months = second_semester_months
    merged_colum_names = []

    for col in columns:
        for month in months:
            column_name = column_name_prefix + col + '_' + str(month)
            merged_colum_names.append(column_name)

    dataframe_transformed_df = pd.DataFrame(index=range(1973, 2018, 1), columns=merged_colum_names)

    for year in range(initial_year, end_year, 1):
        #     print("Year " , year) # FOR DEBUG ONLY
        for month in range(7, 13, 1):
            for column_name in dataframe.columns:
                value = dataframe.loc[str(year) + '-' + str(month) + '-01'][column_name]
                column_name = column_name_prefix + column_name + '_' + str(month)
                dataframe_transformed_df.set_value(year, column_name, value)
    return dataframe_transformed_df


def get_indexes_and_rows_from_second_semester(dataframe):
    indexes = []
    rows = []

    for index, row in dataframe.iterrows():
        if index.month not in second_semester_months:
            continue
        indexes.append(index)
        rows.append(row)
    return indexes, rows


def scalar_product_pws(array_dados_brutos_x, array_dados_brutos_y):
    pws = []
    for month_index in range(len(array_dados_brutos_x)):
        scalar_product_list = []
        for value_index in range(len(array_dados_brutos_x[month_index])):
            scalar_product = math.sqrt(pow(array_dados_brutos_x[month_index][value_index], 2) + pow(
                array_dados_brutos_y[month_index][value_index], 2))
            if scalar_product == 14141.852781018475:
                scalar_product_list.append(np.nan)
            else:
                scalar_product_list.append(scalar_product)
        pws.append(scalar_product_list)
    return pws


def test_knn(X, y, X_test):
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X, y)
    new_pred_class = knn.predict(X_test)

    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, knn


def test_svc(X, y, X_test):
    svc = SVC()
    svc.fit(X, y)
    new_pred_class = svc.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, svc


def test_gpc(X, y, X_test):
    gpc = GaussianProcessClassifier()
    gpc.fit(X, y)
    new_pred_class = gpc.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, gpc


def test_dtc(X, y, X_test):
    dtc = DecisionTreeClassifier(random_state=1)
    dtc.fit(X, y)
    new_pred_class = dtc.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, dtc


def test_rfc(X, y, X_test):
    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(X, y)
    new_pred_class = rfc.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, rfc


def test_mlp(X, y, X_test):
    mlp = MLPClassifier(random_state=1)
    mlp.fit(X, y)
    new_pred_class = mlp.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, mlp


def test_adc(X, y, X_test):
    adc = AdaBoostClassifier(random_state=1)
    adc.fit(X, y)
    new_pred_class = adc.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, adc


def test_gnb(X, y, X_test):
    gnb = GaussianNB()
    gnb.fit(X, y)
    new_pred_class = gnb.predict(X_test)
    year = 2001
    for v in new_pred_class:
        print(year, v)
        year += 1
    return new_pred_class, gnb


def load_sst_data(diretorio_arquivo_geral="funceme_db/tsm/geral/_Dados_TSMvento_2014_04_sst6414b04",
                  diretorio_arquivos_individuais="funceme_db/anomalia_tsm/individual/"):
    array_de_dados_sst = merge_dados_do_diretorio(diretorio_arquivo_geral, diretorio_arquivos_individuais)
    funceme_sst_df = inicia_funceme_data_frame(array_de_dados_sst)
    funceme_sst_df = funceme_sst_df.loc['1973-01-01':'2017-12-01']
    funceme_sst_df = funceme_sst_df.replace(9999.8, np.nan)
    funceme_sst_df = funceme_sst_df.dropna(axis=1, how='any')

    indexes_from_second_semester, rows_from_second_semester = get_indexes_and_rows_from_second_semester(funceme_sst_df)

    sst_full_area_jul_dez_df = pd.DataFrame(index=indexes_from_second_semester, columns=funceme_sst_df.columns,
                                            data=rows_from_second_semester)
    return sst_full_area_jul_dez_df

def load_pws_data():
    array_dados_brutos_x = merge_dados_do_diretorio(
        "funceme_db/pseudo_tensao_x_dados_brutos/geral/_Dados_TSMvento_2014_04_pwsx6414b04",
        "funceme_db/pseudo_tensao_x_dados_brutos/individual/")
    array_dados_brutos_y = merge_dados_do_diretorio(
        "funceme_db/pseudo_tensao_y_dados_brutos/geral/_Dados_TSMvento_2014_04_pwsy6414b04",
        "funceme_db/pseudo_tensao_y_dados_brutos/individual/")

    pws = scalar_product_pws(array_dados_brutos_x, array_dados_brutos_y)
    pws_full_area_df = inicia_funceme_data_frame(pws)

    indexes_from_second_semester, rows_from_second_semester = get_indexes_and_rows_from_second_semester(
        pws_full_area_df)

    pws_full_area_jul_dez_df = pd.DataFrame(index=indexes_from_second_semester, columns=pws_full_area_df.columns,
                                            data=rows_from_second_semester)
    pws_full_area_jul_dez_df = pws_full_area_jul_dez_df.loc['1973-01-01':'2017-12-01']
    pws_full_area_jul_dez_df = pws_full_area_jul_dez_df.replace(14141.852781018475, np.nan)
    pws_full_area_jul_dez_df = pws_full_area_jul_dez_df.dropna(axis=1, how='any')
    return pws_full_area_jul_dez_df

def plot_roc(classifier, X_train, X_test, y_train, y_test, n_classes, algorithm="algoritmo"):
    # learn to predict each class against the other
    one_vs_rest_classifier = OneVsRestClassifier(classifier)
    y_score = 0
    try:
        y_score = one_vs_rest_classifier.fit(X_train, y_train).predict_proba(X_test)
    except AttributeError:
        try:
            y_score = one_vs_rest_classifier.fit(X_train, y_train).decision_function(X_test)
        except AttributeError:
            print("AttributeError on decision_function")


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    names=['fraco','normal','forte']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Classe {0} (area = {1:0.2f})'.format(names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de falso positivo')
    plt.ylabel('Taxa de verdadeiro positivo')
    plt.title('Curva ROC para o algoritmo ' + algorithm)
    plt.legend(loc="lower right")
    plt.show()