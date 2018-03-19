baixar_media_macroregiao(){
    echo $1
    curl -d "produto=macrorregioes&metrica=media&periodo=mensal&data=2018-2-17&mes=2&ano=$1&decada=2018-2018" -H "Content-Type: application/x-www-form-urlencoded" -X POST http://funceme.br/app/calendario/produto/macrorregioes/media/mensal > $1.html
}

for i in {2005..2018}
do
    baixar_media_macroregiao $i
done