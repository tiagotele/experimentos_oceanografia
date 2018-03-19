from bs4 import BeautifulSoup

if __name__ == '__main__':

    csv_header = "datahora,Macrorregioes,Normal(mm),Observado(mm),Desvio(%)"
    csv_lines = []

    for current_year in range(2005, 2018):
        year = str(current_year)
        day = "01"
        month = ""
        default_time = " 12:00:00+00:00"

        raw_html = open('../scraping/' + year + '.html').read()
        html = BeautifulSoup(raw_html, 'html.parser')
        tbody = html.findAll("tbody")
        for tbody_index in range(len(tbody)):
            month = "0"+str(tbody_index + 1) if tbody_index < 9 else str(tbody_index + 1)
            for tr in tbody[tbody_index].findAll("tr"):
                linha = []
                for td in tr:
                    if td == "\n":
                        continue
                    linha.append(td.text)
                csv_lines.append(year + "-" + month + "-" + day +  default_time + "," + ",".join(linha))

    with open('funceme_media_macrorregiao.csv', 'a') as csv_funceme:
        csv_funceme.write(csv_header)
        for csv_line in csv_lines:
            csv_funceme.write("\n")
            csv_funceme.write(csv_line)
