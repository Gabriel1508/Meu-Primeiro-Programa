
#Programa que identifica altas repentinas nos preços das moedas que o robo trade ja comprou

                                  #Bibliotecas

import os.path
import pandas as pd
from binance.client import Client #Biblioteca da Binance
from datetime import datetime
                                    #Funções

def minutos_da_nova_data(data_inicio, simbolo, intervalo, data, corretora):
    
    '''
     Esta função tem como objetivo converter um determinado intervalo selecionado para minutos. Por exemplo:
    1h = 60m, 1d = 1440m
    
    https://medium.com/better-programming/easiest-way-to-use-the-bitmex-api-with-python-fbf66dc38633
    '''
    
    if corretora == 'binance':
        old = datetime.strptime(data_inicio, '%d %b %Y') #Ponto inicial(primeiro dia, por exemplo) da base de dados extraídas da Binance
        new = pd.to_datetime(binance_client.get_klines(symbol=simbolo, interval=intervalo)[-1][0], unit='ms') #Ponto final(ultimo dia, por exemplo) da base de dados extraídas da Binance
        
    return old, new

def data_binance(data_inicio, simbolo, intervalo):

    '''
     Esta função extrai todos os dados que queremos da corretora Binance, especialmente a data que contém
    os preços de abertura(Open), fechamento(Close), máximos(High), mínimos(Low) e volume(Volume). Além da data
    (Date) em que estamos trabalhando
    
    https://medium.com/better-programming/easiest-way-to-use-the-bitmex-api-with-python-fbf66dc38633
    '''
    
    data_df = pd.DataFrame() #Extraíndo da base da dados em formato DataFrame
    
    ponto_antigo, ponto_novo = minutos_da_nova_data(data_inicio, simbolo, intervalo, data_df, corretora = 'binance')
    
    print('Fazendo Download da base no intervalo de %s para o par de cripto %s desde %s.' % (intervalo, simbolo, data_inicio)) #Mensagem enquanto faz download dos dados
        
    dados = binance_client.get_historical_klines(simbolo, intervalo, ponto_antigo.strftime('%d %b %Y %H:%M:%S')
                                                    , ponto_novo.strftime('%d %b %Y %H:%M:%S')) #Esta função, propria da biblioteca binance.client, extrai diretamente os dados que queremos da binance

    data = pd.DataFrame(dados, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av'
                                          , 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms') #Criando coluna timestamp, formada pelas datas da data em formato pandas._libs.tslibs.timestamps.Timestamp
    
    data_df = data #Renomeando a data
    data_df.set_index('timestamp', inplace=True) #Substitui a coluna de índices pela coluna timestamp

    return data_df


def cloSe(base): #S maíusculo para não dar conflito com a função para criação de arquivos .close()
    
    '''
     Esta função extrai os dados de preços de fechamento(Open) de cada Candle(na determinada data) da base.
    '''
    
    Close = base['close'] #Coluna Close, preço final do bitcoin no dia
    Close = Close.astype(float) #Transformar Valores dos valores de cada linha da coluna Close de string para float

    return Close

                                 #Data

#API's extraídos da Binance:

binance_api_key = '' #Coloque seu API KEY dentro das aspas ''
binance_api_secret = '' #Coloque seu APy SECRET KEY dentro das aspas ''

#Intervalos de tempo:

binsizes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60
            , '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720 , '1d': 1440
            , '1w': 10080} #m representa minuto(1 minuto), h hora(60 minutos), d dia(1440 minutos) e w semana(10080)

batch_size = 250
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret) #Função que utiliza os API's da Binance

#Escolhendo os parametros para criar a base:

data_inicio = '26 Nov 2019' #Escolha a Data de início para a criação da base de dados do par de moedas escolhido. Deve ser entre aspas e dias, meses e anos espaçados e os meses abreviados com inicial maiuscula
intervalo = '1d' #Intervalo de tempo, opções encontradas no binsizes acima

#Abaixo segue uma lista de simbolos de pares de moedas listadas na binance. O programa irá
#analisar cada uma delas. Voce pode escolher quantas quiser

binance_simbolos = ['BQXBTC', 'CHZBTC', 'MATICBTC', 'ONTBTC', 'TNTBTC']

vendida = [] #Moedas vendidas

while True:
    for simbolo in binance_simbolos:
        data = data_binance(data_inicio, simbolo, intervalo) #Extraindo a base de dados
        base = data
        base = base.reset_index() #Resetando os índices, para que começem no 0

        #Extraindo os dados da base em forma de colunas:

        Close = cloSe(base)
        
        if os.path.exists('%s.txt' %simbolo) == True: #Caso arquivo exista(moeda tenha sido comprada):
            ler_arquivo = open('%s.txt' %simbolo, 'r').read().split() #Ler primeira linha do arquivo da moeda que foi comprada
            if Close[len(base) - 1] >= 1.10 * (int(ler_arquivo[0]) / 0.01) ** (-1): #Caso a moeda tenha subido 10%, vedemos:
                ordem_venda = binance_client.create_order( #Função que executa ordem de venda na binance
                                           symbol = simbolo, #Simbolo da moeda
                                           side = Client.SIDE_SELL, #Ordem de venda
                                           type = Client.ORDER_TYPE_MARKET, #Vender usando ordem de preço de mercado
                                           quantity = ler_arquivo[0]) #Vendendo a mesma quantidade que compramos
        
                print('{} VENDIDA!'.format(simbolo)) #Mostrar print de que a moeda foi vendida
                os.remove('%s.txt' %simbolo) #Remove o arquivo que foi criado quando foi realizado uma compra
                
                vendida.append(simbolo) #Acumulando as moedas vendidas