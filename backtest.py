
'''

BACKTEST:
    
 Está programado para gráficos de tempos curtos como 15m, 30m, 1h.
Cada ponto do gráfico representa o lucro de algum dia atras. Por exemplo,
o ultimo ponto representa o lucro de hoje, penúltimo representa o lucro de
ontém até hoje, antepenúltimo representa lucro de anteontem até hoje.

'''
                                  #Bibliotecas

import calendar
import os.path
import time
import numpy as np
import pandas as pd
import matplotlib #Usado para gráficos
import matplotlib.pyplot as plt #Usado para plotar gráficos
import warnings
from binance.client import Client #Biblioteca da Binance
from datetime import datetime 
from mpl_finance import candlestick_ochl as candlestick #Usado para criar gráfico de candlesticks
from matplotlib.dates import num2date #Usado em gráficos, muda o formato numérico de datas para o formato padrão(dia, mês, ano)
from matplotlib.dates import date2num #Usado em gráficos, muda o formato da data(dia, mês, ano) para números
from numpy import NaN as npNaN
from numpy import log as nplog
from pandas import Series
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

def date(base):
    
    '''
     Esta função extrai os dados da Data da base.
    '''
    
    Date = base['timestamp'] #Coluna Date, msotrando a data
    return Date

def high(base):
    
    '''
     Esta função extrai os dados de preços máximos(High) de cada Candle(na determinada data) da base.
    '''
    
    High = base['high'] #Coluna High, o maior preço que o bitcoin atingiu no dia
    High = High.astype(float) #Transformar Valores dos valores de cada linha da coluna High de string para float

    return High

def low(base):
    
    '''
     Esta função extrai os dados de preços mínimos(Low) de cada Candle(na determinada data) da base.
    '''
    
    Low = base['low'] #Coluna Low, o menor preço que o bitcoin atingiu no dia
    Low = Low.astype(float) #Transformar Valores dos valores de cada linha da coluna Low de string para float

    return Low

def opeN(base): #N maíusculo para não dar conflito com a função para criação de arquivos .open()
    
    '''
     Esta função extrai os dados de preços de abertura(Open) de cada Candle(na determinada data) da base.
    '''
    
    Open = base['open'] #Coluna Open, preço inicial do bitcoin no dia
    Open = Open.astype(float) #Transformar Valores dos valores de cada linha da coluna Open de string para float

    return Open

def cloSe(base): #S maíusculo para não dar conflito com a função para criação de arquivos .close()
    
    '''
     Esta função extrai os dados de preços de fechamento(Open) de cada Candle(na determinada data) da base.
    '''
    
    Close = base['close'] #Coluna Close, preço final do bitcoin no dia
    Close = Close.astype(float) #Transformar Valores dos valores de cada linha da coluna Close de string para float

    return Close

def volume(base):
    
    '''
     Esta função extrai o Volume(nas determinadas datas) da base.
    '''
    
    Volume = base['volume'] #Coluna Volume, mostra o volume das transações de bitcoin
    Volume = Volume.astype(float) #Transformar Valores dos valores de cada linha da coluna Volume de string para float

    return Volume

def preco(High, Low, Close):
    
    '''
     Definiremos agora a coluna de Preço típico como sendo a média entre o preço mais alto(High), o preço
    mais baixo(Low) e o preço de fechamento(Close) de um determinado dia.
    '''  
    
    Preco = (High + Low + Close) / 3.0 #Preço médio(ou preço típico) do Bitcoin em relação ao dolar
    return Preco

def cci(periodoCCI, constante, Preco):
    
    '''
     Commodity Channel Index(CCI)
    
    https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci
    https://www.investopedia.com/terms/c/commoditychannelindex.asp    
    '''
    
    precoSMA = Close.rolling(window = periodoCCI).mean() #SMA de Close de períodoCCI

    desvio_medioCCI = pd.DataFrame(np.zeros((len(base),1))) #Coluna em DataFrame contendo valores nulos que posteriormente será preenchida por valores do desvio médio do CCI
    desvio_medioCCI = desvio_medioCCI.squeeze() #Passando a coluna criada acima de DataFrame para series

    #No for abaixo preenchemos a coluna criada acima com valores do desvio médio:

    j = 1
    for i in range(periodoCCI,len(base)):
        while j <= periodoCCI:
            desvio_medioCCI[i] += abs(precoSMA[i] - Close[i - j + 1]) #O cálculo é feito pelo somatório do módulo da diferença entre a SMA do Preço no períodoCCI e os valores anteriores(períodoCCI) do Preço
            j += 1
        j = 1
    
    #E no final tudo dividido pelo número de termos:
    
    desvio_medioCCI = desvio_medioCCI / periodoCCI 

    #Fómulas finais, onde a constante(geralmente igual a 0.015) que serve para deixar cerca de 70% - 80% dos valores na faixa de +- 100%

    CCI = (Close - precoSMA) / (constante * desvio_medioCCI) #Fórmula final do Woodies CCI

    return CCI

def rsi(periodoRSI, Close):
    
    '''
     Relative Strength Index
    
    https://en.wikipedia.org/wiki/Relative_strength_index
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI
    https://www.investopedia.com/terms/r/rsi.asp
    '''
    
    CLOSE = Close.values.tolist()
    lista_rsi = []
    media_ganhos = .0
    media_percas = .0
    i = 0
    
    while i < len(Close):
        if i + 1 < periodoRSI: #Corrigindo as primeiras posições menores que o número do período escolhido
            rsi = float('NaN')
        else:
            if CLOSE[i] > CLOSE[i - 1]:
                up = Close[i] - Close[i - 1] #Dias de ganhos
                dn = 0 #0 para perdas
            else:
                up = 0 #0 para ganhos
                dn = CLOSE[i - 1] - CLOSE[i] #Dias de perdas
            media_ganhos = (media_ganhos * (periodoRSI - 1) + up) / periodoRSI #Média de ganhos utilizando a RMA(https://en.wikipedia.org/wiki/Moving_average#Modified_moving_average)
            media_percas = (media_percas * (periodoRSI - 1) + dn) / periodoRSI #Média de perdas utilizando a RMA(https://en.wikipedia.org/wiki/Moving_average#Modified_moving_average)
            if media_ganhos + media_percas == 0 and media_ganhos > 0:
                rsi = 100
            elif media_ganhos + media_percas == 0 and media_ganhos == 0:
                rsi = 0
            else:
                rsi = 100 * media_ganhos / (media_ganhos + media_percas) #Fórmula final
        lista_rsi.append(rsi) #Juntando valores anteriores na lista_rsi
        i += 1
        
    lista_rsi = pd.DataFrame(lista_rsi)
    lista_rsi = lista_rsi.squeeze()
    
    return lista_rsi
    
    return RSI

def uo(periodoUO_1, periodoUO_2, periodoUO_3, pesoUO_1, pesoUO_2, pesoUO_3, Low, High, Preco, Close):
    
    '''
     Ultimate Oscillator(UO)
     
    https://www.investopedia.com/terms/u/ultimateoscillator.asp
    https://en.wikipedia.org/wiki/Ultimate_oscillator
    https://blog.iqoption.com/pt/como-negociar-com-o-ultimate-oscillator-um-guia-passo-a-passo-para-iniciantes/
    '''
    
    menor_close_low = pd.concat([Close.shift(1), Low], axis = 1) #Juntando as colunas Close anterior e Low atual
    menor_close_low = menor_close_low.min(axis = 1) #Menor valor entre Close anterior e Low atual

    maior_close_high = pd.concat([Close.shift(1),High], axis = 1) #Juntando as colunas Close anterior e High atual
    maior_close_high = maior_close_high.max(axis = 1) #Maior valor entre Close anterior e High atual

    pressao_compra = Close - menor_close_low #Definindo pressão de compra
    alcance_verdadeiro = maior_close_high - menor_close_low #Definindo alcançe verdadeiro

    #Abaixo as médias de três períodos diferentes, calculada pela divisão entre o somatório da pressão de
    #compra em um determinado período e pelo somatório do alcance verdadeiro no mesmo período:

    media7 = pressao_compra.rolling(periodoUO_1, min_periods=0).sum() / alcance_verdadeiro.rolling(periodoUO_1, min_periods=0).sum()
    media14 = pressao_compra.rolling(periodoUO_2, min_periods=0).sum() / alcance_verdadeiro.rolling(periodoUO_2, min_periods=0).sum()
    media28 = pressao_compra.rolling(periodoUO_3, min_periods=0).sum() / alcance_verdadeiro.rolling(periodoUO_3, min_periods=0).sum()

    #Fórmula final:

    Ult_Osc = 100 * ((pesoUO_1 * media7) + (pesoUO_2 * media14) + (pesoUO_3 * media28)) / (pesoUO_1 + pesoUO_2 + pesoUO_3) #Fórmula final

    return Ult_Osc

def osc_est(periodoEstocastico, Low, High, Close):
    
    '''
     Stochastic Oscillator, ou oscilador estocástico, mede a velocidade e momento do preço afim de prever
    reversões de Preço
    
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    https://en.wikipedia.org/wiki/Stochastic_oscillator
    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full
    '''

    low_minimo14 = Low.rolling(window = periodoEstocastico, min_periods=0).min() #Menor valor de Low no período estocastico
    high_maximo14 = High.rolling(window = periodoEstocastico, min_periods=0).max() #Maior valor de High no período estocastico
    osc_est = 100 * (Close - low_minimo14) / (high_maximo14 - low_minimo14) #Fórmula final

    return osc_est

def williamr(periodoWilliams, High, Low):
    
    '''
     Williams %R
     
    https://www.bussoladoinvestidor.com.br/williams-r-indicador-de-momento/
    https://br.advfn.com/educacional/analise-tecnica/william-percent-range
    https://www.ifcmarkets.com.br/ntx-indicators/williams-percent-range
    '''
    
    maior_high = High.rolling(periodoWilliams, min_periods=0).max() #Maior valor de High no período Williams
    menor_low = Low.rolling(periodoWilliams, min_periods=0).min() #Menor valor de low no período Williams

    William_R = -100 * (maior_high - Close) / (maior_high - menor_low)  #Fórmula final do Williams %R

    return William_R

def demarker(periodoMarker, High, Low):
    
    '''
     DeMarker
     
    https://www.metatrader5.com/pt/terminal/help/indicators/oscillators/demarker
    https://www.ifcmarkets.com.br/ntx-indicators/demarker
    https://www.investopedia.com/terms/d/demarkerindicator.asp
    '''

    #Criando as coluna De_Max e De_Min de tamanho len(base) de entradas 0 e as transformando de DataFrame para Series

    De_Max = pd.DataFrame(np.zeros((len(base),1)))
    De_Max = De_Max.squeeze()
    De_Min = pd.DataFrame(np.zeros((len(base),1)))
    De_Min = De_Min.squeeze()

    for i in range(1,len(base)):
        if High[i] > High[i - 1]:
            De_Max[i] = High[i] - High[i - 1] #Definição de De_Max que recebe valor apenas quando High de i é maior de High de 1 índice a menos(i - 1)
        else:
            De_Max[i] = 0
        if Low[i] < Low[i - 1]:
            De_Min[i] = Low[i - 1] - Low[i] #Definição de De_Min que recebe valor apenas quando Low de i é maior de Low de 1 índice a menos(i - 1)
        else:
            De_Min[i] = 0
        
    De_Marker = (De_Max.rolling(window = periodoMarker).mean() #Fórmula final do DeMarker
                / (De_Max.rolling(window = periodoMarker).mean() + De_Min.rolling(window = periodoMarker).mean()))
    
    return De_Marker

def cm_rsi_ema(lenCMRSI_EMA1, lenCMRSI_EMA2, Close):
    
    '''
     CM_RSI plus EMA
     
    https://www.tradingview.com/script/LZGJt8sd-CM-RSI-Plus-EMA/
    '''
    
    #Criando as coluna up, down, MaiorCloseDif, MenorCloseDif de tamanhos len(base) com entradas 0 e as transformando de DataFrame para Series
    
    up = pd.DataFrame(np.zeros((len(base),1)))
    up = up.squeeze()
    down = pd.DataFrame(np.zeros((len(base),1)))
    down = down.squeeze()
    MaiorCloseDif = pd.DataFrame(np.zeros((len(base),1)))
    MaiorCloseDif = MaiorCloseDif.squeeze()
    MenorCloseDif = pd.DataFrame(np.zeros((len(base),1)))
    MenorCloseDif = MenorCloseDif.squeeze()
    
    for i in range(1,len(base)):
        if Close[i] > Close[i - 1]:
            MaiorCloseDif[i] = Close[i] - Close[i - 1] #MaiorCloseDif recebe valor apenas quando Close i é maior que Close 1 índice a menos(i - 1)
            MenorCloseDif[i] = 0
        elif Close[i] < Close[i - 1]:
            MaiorCloseDif[i] = 0
            MenorCloseDif[i] = Close[i] - Close[i - 1] #MenorCloseDif recebe valor apenas quando Close i é menor que Close 1 índice a menos(i - 1)
    
    for i in range(1,len(base)):
        up[i] = MaiorCloseDif[i] * 1 / lenCMRSI_EMA1  + up[i - 1] * (1 - 1 / lenCMRSI_EMA1)
        down[i] = -MenorCloseDif[i] * 1 / lenCMRSI_EMA1 + down[i - 1] * (1 - 1 / lenCMRSI_EMA1)
    
    CMRSI = 100 - (100 / (1 + up / down))
    EMACRSI = CMRSI.ewm(span = lenCMRSI_EMA2, adjust = False).mean()

    return CMRSI, EMACRSI

def bb_b(periodoBB_B, numero_desvio_padrao_BB_B, Close):
    
    '''
    Bolinger Bands %B
    
    https://www.tradingview.com/wiki/Bollinger_Bands_%25B_(%25B)
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce
    https://www.investopedia.com/terms/b/bollingerbands.asp
    '''
    
    media_BB = Close.rolling(window = periodoBB_B).mean() #SMA de periodoBB_B do Close
    desvio_padrao_BB = Close.rolling(window = periodoBB_B).std() #Desvio padrão do Close num intervalo periodoBB_B
    BB_superior = media_BB + numero_desvio_padrao_BB_B * desvio_padrao_BB #Definição da Banda de Bollinger superior
    BB_inferior = media_BB - numero_desvio_padrao_BB_B * desvio_padrao_BB #Definição da Banda de Bollinger inferior
    
    BB_B = (Close - BB_inferior) / (BB_superior - BB_inferior) #Fórmula final do Bolinger Bands %B
    
    return BB_B

def tsi(periodoTSImaior, periodoTSImenor, Close):

    '''
     True strength index(TSI)
     
    https://en.wikipedia.org/wiki/True_strength_index
    https://www.investopedia.com/terms/t/tsi.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    '''
    
    m = Close.diff() #Coluna Close - Close.shift(1)(Close de 1 índice anterior)
    EMA25_m = m.ewm(span = periodoTSImaior).mean() #EMA de periodoTSImaior de m
    EMA13_EMA25_m = EMA25_m.ewm(span = periodoTSImenor).mean() #EMA de periodoTSImenor do EMA25_m
    EMA25_abs_m = (abs(m)).ewm(span = periodoTSImaior).mean() #EMA de periodoTSImaior do módulo de m
    EMA13_EMA25_abs_m = EMA25_abs_m.ewm(span = periodoTSImenor).mean() #EMA de periodoTSImenor do EMA25_absm
    
    TSI = 100 * EMA13_EMA25_m / EMA13_EMA25_abs_m #Fórmula final do TSI

    return TSI

def smie(periodoSMIElongo, periodoSMIEcurto, periodoSMIEsinal, Close):

    '''
     Stochastic Momentum Index(SMI) Ergodic
     
    https://br.tradingview.com/script/Xh5Q0une-SMI-Ergodic-Oscillator/
    '''
    
    CloseDif = Close.diff() #Coluna Close - Close.shift(1)(Close de 1 índice anterior)
    CloseDif_abs = abs(CloseDif) #Módulo de CloseDif
    CloseDif_SMA = CloseDif.ewm(span = periodoSMIEcurto, adjust = False).mean() #EMA de periodoSMIEcurto de CloseDif
    CloseDif_abs_SMA = CloseDif_abs.ewm(span = periodoSMIEcurto, adjust = False).mean() #EMA de periodoSMIEcurto de CloseDif_abs
    CloseDif_SMA_SMA = CloseDif_SMA.ewm(span = periodoSMIElongo, adjust = False).mean() #EMA de periodoSMIElongo de CloseDif_SMA
    CloseDif_abs_SMA_SMA = CloseDif_abs_SMA.ewm(span = periodoSMIElongo, adjust = False).mean() #EMA de periodoSMIElongo de CloseDif_abs_SMA
    
    SMIE = CloseDif_SMA_SMA / CloseDif_abs_SMA_SMA #Fórmula final do SMI Ergodic
    SMIE_sinal = SMIE.ewm(span = periodoSMIEsinal).mean()
    
    return SMIE, SMIE_sinal

def fisher(High, Low, length=None):
    
    '''
     Fisher Transform Indicator
     
    https://www.tradingview.com/script/og7JPrRA-CM-Williams-Vix-Fix-Finds-Market-Bottoms/
    '''  

    length = int(length) if length and length > 0 else 5

    # Calculate Result
    m = High.size
    hl2_ = (High + Low) / 2
    max_high = hl2_.rolling(length).max()
    min_low = hl2_.rolling(length).min()
    hl2_range = max_high - min_low
    hl2_range[hl2_range < 1e-5] = 0.001
    position = (hl2_ - min_low) / hl2_range
    
    v = 0
    fish = 0
    result = [npNaN for _ in range(0, length - 1)]
    for i in range(length - 1, m):
        v = 0.66 * (position[i] - 0.5) + 0.67 * v
        if v >  0.99: v =  0.999
        if v < -0.99: v = -0.999
        fish = 0.5 * (fish + nplog((1 + v) / (1 - v)))
        result.append(fish)
        
    fisher = Series(result)

    return fisher

def rvi(comprimentoRVI_revisao, comprimentoRVI_original, High, Close):
    
    '''
     Relative Volatility Index(RVI) Original e Relative Volatility Index(RVI) Revision
    '''
    
    #Abaixo a criação das colunas upH, dnH, upL, dnL, upC, dnC de tamanho len(base) e convertidas de DataFrame para Series
    
    upH = pd.DataFrame(np.zeros((len(base),1)))
    upH = upH.squeeze()
    dnH = pd.DataFrame(np.zeros((len(base),1)))
    dnH = dnH.squeeze()
    upL = pd.DataFrame(np.zeros((len(base),1)))
    upL = upL.squeeze()
    dnL = pd.DataFrame(np.zeros((len(base),1)))
    dnL = dnL.squeeze()
    upC = pd.DataFrame(np.zeros((len(base),1)))
    upC = upC.squeeze()
    dnC = pd.DataFrame(np.zeros((len(base),1)))
    dnC = dnC.squeeze()
    
    for i in range(9,len(base)):
        if High[i] > High[i - 1]:
            upH[i] = High[i - 9:i].std() #Caso High no índice i seja maior que High no índice anterior (i - 1), upH é definido pelo desvio padrão do High no intervalo de (i - 9) até i
            dnH[i] = 0
        elif High[i] < High[i - 1]:
            upH[i] = 0
            dnH[i] = High[i - 9:i].std() #Caso High no índice i seja menor que High no índice anterior (i - 1), dnH é definido pelo desvio padrão do High no intervalo de (i - 9) até i
        if Low[i] > Low[i - 1]:
            upL[i] = Low[i - 9:i].std() #Caso Low no índice i seja maior que Low no índice anterior (i - 1), upL é definido pelo desvio padrão do High no intervalo de (i - 9) até i
            dnL[i] = 0
        elif Low[i] < Low[i - 1]:
            upL[i] = 0
            dnL[i] = Low[i - 9:i].std() #Caso Low no índice i seja menor que Low no índice anterior (i - 1), dnL é definido pelo desvio padrão do High no intervalo de (i - 9) até i
        if Close[i] > Close[i - 1]:
            upC[i] = Close[i - 9:i].std() #Caso Close no índice i seja maior que Close no índice anterior (i - 1), upC é definido pelo desvio padrão do High no intervalo de (i - 9) até i
            dnC[i] = 0
        elif Close[i] < Close[i - 1]:
            upC[i] = 0
            dnC[i] = Close[i - 9:i].std() #Caso Close no índice i seja menor que Close no índice anterior (i - 1), dnC é definido pelo desvio padrão do High no intervalo de (i - 9) até i

    #Abaixo a criação das colunas upavgH, dnavgH, upavgL, dnavgL, upavgC, dnavgC de tamanho len(base) e convertidas de DataFrame para Series
            
    upavgH = pd.DataFrame(np.zeros((len(base),1)))
    upavgH = upavgH.squeeze()
    dnavgH = pd.DataFrame(np.zeros((len(base),1)))
    dnavgH = dnavgH.squeeze()
    upavgL = pd.DataFrame(np.zeros((len(base),1)))
    upavgL = upavgL.squeeze()
    dnavgL = pd.DataFrame(np.zeros((len(base),1)))
    dnavgL = dnavgL.squeeze()
    upavgC = pd.DataFrame(np.zeros((len(base),1)))
    upavgC = upavgC.squeeze()
    dnavgC = pd.DataFrame(np.zeros((len(base),1)))
    dnavgC = dnavgC.squeeze()
    
    #Abaixo a definição das colunas upavgH, dnavgH, upavgL, dnavgL, upavgC, dnavgC, que utilizam a Modified moving average(https://en.wikipedia.org/wiki/Moving_average#Modified_moving_average)
    
    for i in range(1,len(base)):
        upavgH[i] = (upavgH[i - 1] * (comprimentoRVI_revisao - 1) + upH[i]) / comprimentoRVI_revisao
        dnavgH[i] = (dnavgH[i - 1] * (comprimentoRVI_revisao - 1) + dnH[i]) / comprimentoRVI_revisao
        upavgL[i] = (upavgL[i - 1] * (comprimentoRVI_revisao - 1) + upL[i]) / comprimentoRVI_revisao
        dnavgL[i] = (dnavgL[i - 1] * (comprimentoRVI_revisao - 1) + dnL[i]) / comprimentoRVI_revisao
        upavgC[i] = (upavgC[i - 1] * (comprimentoRVI_original - 1) + upC[i]) / comprimentoRVI_original
        dnavgC[i] = (dnavgC[i - 1] * (comprimentoRVI_original - 1) + dnC[i]) / comprimentoRVI_original
        
    RVIH = 100 * upavgH / (upavgH + dnavgH)
    RVIL = 100 * upavgL / (upavgL + dnavgL)
    
    RVI_original = 100 * upavgC / (upavgC + dnavgC) #Fórmula final do RVI Original
    
    RVI_revisao = (RVIH + RVIL) / 2 #Fórmula final do RVI Revision

    return RVI_revisao, RVI_original

def cm_williams_vix_fix(LookBackstd, comprimentoBB, BBstd, lookBackPercentil, maiorpercentil,
                        menorpercentil, Close):

    '''
     CM_Williams_Vix_Fix
     
    https://www.tradingview.com/script/og7JPrRA-CM-Williams-Vix-Fix-Finds-Market-Bottoms/
    '''
    
    WVF = 100 * (Close.rolling(window = LookBackstd).max() - Low) / Close.rolling(window = LookBackstd).max() #Williams Vix Fix é definido pelo maior Close no intervalo de período LookBackstd menos Low. A partír desta diferença, dividir pelo maior Close no intervalo de período LookBackstd
    WVF_Std = BBstd * WVF.rolling(window = comprimentoBB).std() #Desvio padrão do WVF num intervalo de período comprimentoBB
    
    WVF_linha_meio = WVF.rolling(window = comprimentoBB).mean() #Média do WVF num intervalo de período comprimentoBB
    #WVF_linha_inferior = WVF_linha_meio - WVF_Std
    WVF_linha_superior = WVF_linha_meio + WVF_Std
    
    WVF_alcancemaior = maiorpercentil * WVF.rolling(window = lookBackPercentil).max() #Maior valor de WVF num intervalo de período lookBackPercentil
    #WVF_alcancemenor = menorpercentil * WVF.rolling(window = lookBackPercentil).min()

    return WVF, WVF_linha_superior, WVF_alcancemaior

def imi(periodoIMI, Close, Open):

    '''
     Intraday Momentum Index(IMI)]
    
    https://www.investopedia.com/terms/i/intraday-momentum-index-imi.asp
    https://www.technicalindicators.net/indicators-technical-analysis/173-imi-intraday-momentum-index
    https://library.tradingtechnologies.com/trade/chrt-ti-intraday-momentum-index.html
    '''
    
    GanhosSMI = pd.DataFrame(np.zeros((len(base),1))) #Criando coluna GanhosSMI de tamanho len(base) formada por zeros
    GanhosSMI = GanhosSMI.squeeze() #Passando de DataFrame para series a coluna GanhosSMI
    PerdasSMI = pd.DataFrame(np.zeros((len(base),1)))
    PerdasSMI = PerdasSMI.squeeze()
    
    for i in range(1,len(base)):
        if Close[i] > Open[i]:
            GanhosSMI[i] = Close[i] - Open[i] #Caso Close seja maior que Open no determinado índice i, definimos o GanhosSMI pela fórmula definida
        elif Close[i] < Open[i]:
            PerdasSMI[i] = Open[i] - Close[i] #Caso Open seja maior que Close no determinado índice i, definimos o PerdasSMI pela fórmula definida
            
    IMI = (100 * (GanhosSMI.rolling(window = periodoIMI).sum() #Fórmula final do Intraday Momentum Index
            / (GanhosSMI.rolling(window = periodoIMI).sum() + PerdasSMI.rolling(window = periodoIMI).sum())))
    
    return IMI

def stochrsi(periodoStochRSI, peridostochrsi_sinal):

    '''
     StochRSI
     
    https://www.investopedia.com/terms/s/stochrsi.asp
    https://www.binance.vision/pt/economics/stochastic-rsi-explained
    '''
    
    MenorRSI = RSI.rolling(window = periodoStochRSI).min() #Menor valor do RSI no intervalo de períodoStichRSI
    MaiorRSI = RSI.rolling(window = periodoStochRSI).max() #Maior valor do RSI no intervalo de períodoStichRSI
    
    StochRSI = 100 * (RSI - MenorRSI) / (MaiorRSI - MenorRSI) #Fórmula final do Stoch Relative Strength Index

    k = StochRSI.rolling(window = peridostochrsi_sinal).mean()
    d = k.rolling(window = peridostochrsi_sinal).mean()
    
    return k, d

def smi(periodoSMI, periodoLinhaSinalSMI, High, Low, Close):
    
    '''
    (21)
     SMI
     
    https://www.tradingview.com/script/HLbqdCku-Stochastic-Momentum-Index-SMI/
    https://www.investopedia.com/ask/answers/021315/what-difference-between-stochastic-oscillator-stochastic-momentum-index.asp
    https://www.marketbeat.com/financial-terms/what-is-stochastic-momentum-index/
    '''

    hh = High.rolling(window = periodoSMI).max() #Maior valor de High no intervalo períodoSMI
    ll = Low.rolling(window = periodoSMI).min() #Menor valor de Low no intervalo períodoSMI

    C = (hh + ll) / 2
    
    h = Close - C

    HS1 = h.ewm(span = 3, adjust = False).mean() #EMA de período 3(EMA3) de h
    HS2 = HS1.ewm(span = 3, adjust = False).mean() #EMA3 de HS1

    DHL1 = (hh - ll).ewm(span = 3).mean() #EMA3 de (hh - ll)
    DHL2 = DHL1.ewm(span = 3).mean() #EMA3 de DHL1
    DHL2 = DHL2 / 2

    SMI = 100 * HS2 / DHL2 #Fórmula final do Stochastic Momentum Index

    linha_sinal_SMI = SMI.ewm(span = periodoLinhaSinalSMI).mean() #Fórmula final da linha de sinal do Stochastic Momentum Index

    return SMI, linha_sinal_SMI

def wt(periodoWT1, periodoWT2, constanteWT, Preco):
   
    '''
     Wave Trend Oscillator
     
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
    '''
    
    PrecoEMA10 = Preco.ewm(span = periodoWT1, adjust = False).mean() #Calculando EMA de períodoWT1(EMA10) do Preço típico
    Preco_PrecoEMA10 = (abs(Preco - PrecoEMA10)).ewm(span = periodoWT1, adjust = False).mean() #Calculando EMA10 do modulo do (Preco - PrecoEMA10)
    CI = (Preco - PrecoEMA10) / (constanteWT * Preco_PrecoEMA10)
    
    WT = CI.ewm(span = periodoWT2, adjust = False).mean() #Fórmula final

    return WT

def stoch_vx3(longVX3, shortVX3, periodoPRO, High, Low):

    '''
     Stoch_VX3
    
    https://www.tradingview.com/script/xAPlG1uS-Stoch-VX3/
    '''
        
    def L(g): #Valor de g deve ser uma constante
        
        hl2 = (High + Low) / 2
        
        #Criando quatro colunas de tamanho len(base) de entradas 0 e formato Series
        
        L0 = pd.DataFrame(np.zeros((len(base),1)))
        L0 = L0.squeeze()
        L1 = pd.DataFrame(np.zeros((len(base),1)))
        L1 = L1.squeeze()
        L2 = pd.DataFrame(np.zeros((len(base),1)))
        L2 = L2.squeeze()
        L3 = pd.DataFrame(np.zeros((len(base),1)))
        L3 = L3.squeeze()
        
        for i in range(1,len(base)):
            L0[i] = (1 - g) * hl2[i] + g * L0[i - 1]
            L1[i] = -g * L0[i] + L0[i - 1] + g * L1[i - 1]
            L2[i] = -g * L1[i] + L1[i - 1] + g * L2[i - 1]
            L3[i] = -g * L2[i] + L2[i - 1] + g * L3[i - 1]
    
        f = (L0 + 2 * L1 + 2 * L2 + L3) / 6 #Média ponderada
        
        return f
    
    lmas = L(shortVX3) #Função L(g) para g = shortVX3
    lmal = L(longVX3) #Função L(g) para g = longVX3
    
    ppoT = 100 * (lmas - lmal) / lmal #Fórmula do PRO-Top
    
    pctRankT = pd.DataFrame(np.zeros((len(base),1))) #Coluna de tamanho len(base) e entradas 0
    pctRankT = pctRankT.squeeze() #Transformando de DataFrame para Series

    for i in range(periodoPRO,len(base)):
        pctT = ppoT[i - periodoPRO + 1:i + 1].rank(pct = True) #Pegando a coluna ppoT no intevalo entre (i - periodoPRO + 1) e (i + 1) e transformando os valores para porcentagem
        pctRankT[i] = pctT[i] #Utilizando o ultimo valor da coluna pctT
    
    return pctRankT


def cctbbo(periodoCCTBBO, periodoCCTBBOSinal, Close):
    
    '''
     CCT Bollinger Band Oscillator
    
    https://www.tradingview.com/script/iA4XGCJW-CCT-Bollinger-Band-Oscillator/
    '''

    cctbbo_std = Close.rolling(window = periodoCCTBBO).std() #Desvio padrão de Closenum intervalo de periodoCCTBBO
    cctbbo_sma = Close.rolling(window = periodoCCTBBO).mean() #SMA de períodoCCTBBO de Close
    
    CCTBBO = 100 * (Close + 2 * cctbbo_std - cctbbo_sma) / (4 * cctbbo_std) #Fórmula final do CCTBBO
    CCTBBO_sinal = CCTBBO.ewm(span = periodoCCTBBOSinal).mean()
    
    return CCTBBO, CCTBBO_sinal

def cmo(periodocmo, Close):

    '''
     Chande Momentum Oscillator(CMO)
     
    https://www.tradingview.com/script/ogmWth5h-Chande-Momentum-Oscillator/
    '''     
        
    close_dif_maior = Close.diff()
    close_dif_menor = Close.diff()
    
    for i in range(1,len(base)):
        if close_dif_maior[i] < 0:
            close_dif_maior[i] = 0
        if close_dif_menor[i] > 0:
            close_dif_menor[i] = 0
    
    close_dif_menor = -close_dif_menor
         
    upsum = close_dif_maior.rolling(window = periodocmo).sum()
    downsum = close_dif_menor.rolling(window = periodocmo).sum()
       
    cmo = 100 *(upsum - downsum) / (upsum + downsum)
    
    return cmo

def cm_stock_mtf(periodok, periodod):
    
    '''
     CM Stochastic Multi-TimeFrame
     
    https://www.tradingview.com/script/Wylw98ue-CM-Stochastic-Multi-TimeFrame/
    '''
    
    k = Osc_Est.rolling(window = periodok).mean()
    d = k.rolling(window = periodod).mean()
    
    return d, k

def macd(periodomacd_curto, periodomacd_longo, periodomacd_sinal, Close):
    ema14_macd = Close.rolling(window = periodomacd_curto).mean()
    ema26_macd = Close.rolling(window = periodomacd_longo).mean()
    
    macd = ema14_macd - ema26_macd
    macd_sinal = macd.rolling(window = periodomacd_sinal).mean()
    
    return macd, macd_sinal

                                 #Data

#API's extraídos da Binance:
#(Para rodar este programa não é necessário digitar as API KEYS)

binance_api_key = '' #Coloque seu API KEY dentro das aspas ''
binance_api_secret = '' #Coloque seu APy SECRET KEY dentro das aspas ''

#Intervalos de tempo:

binsizes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60
            , '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720 , '1d': 1440
            , '1w': 10080} #m representa minuto(1 minuto), h hora(60 minutos), d dia(1440 minutos) e w semana(10080)

batch_size = 250
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret) #Função que utiliza os API's da Binance

#Lista de datas que serão usadas para o teste do programa:

tempolocal = time.localtime(time.time()) #Mostra a data de hoje(dia, mês, ano, horario, ...)

lista_data = ['15 Nov 2019', '16 Nov 2019', '17 Nov 2019', '18 Nov 2019', '19 Nov 2019'
              , '20 Nov 2019', '21 Nov 2019', '22 Nov 2019', '23 Nov 2019', '24 Nov 2019'
              , '25 Nov 2019', '26 Nov 2019', '27 Nov 2019', '28 Nov 2019', '29 Nov 2019'] 

#Lista de intervalos que serão utilizados para o teste do programa:

lista_intervalo = ['15m', '30m', '1h'] 

#Lista de simbolos da Binance que serão utilizados para o teste do programa:

binance_simbolos_BTC = ['TNTBTC', 'BQXBTC', 'CHZBTC', 'QTUMBTC', 'ZILBTC', 'RVNBTC' 
                    , 'KAVABTC', 'VIBBTC', 'MATICBTC', 'OMGBTC', 'MCOBTC'
                    , 'ADABTC', 'ALGOBTC', 'ATOMBTC', 'BATBTC', 'ONTBTC', 'XTZBTC'
                    , 'IOTABTC', 'KEYBTC', 'STXBTC', 'ONEBTC'
                    , 'DLTBTC', 'KMDBTC', 'EVXBTC', 'ZECBTC', 'BCHABCBTC', 'DATABTC']

binance_simbolos_USDT = ['LTCUSDT', 'LINKUSDT', 'NEOUSDT', 'ADAUSDT', 'BCHABCUSDT', 'XRPUSDT'
                         , 'TRXUSDT', 'BATUSDT', 'EOSUSDT', 'BNBUSDT', 'VETUSDT', 'MATICUSDT'
                         , 'BTCUSDT', 'ETCUSDT', 'ATOMUSDT', 'FETUSDT', 'ONTUSDT', 'XLMUSDT']

binance_simbolos_USANDO = ['MATICBTC', 'MATICUSDT', 'WPRBTC', 'DGDBTC', 'HOTBTC', 'NEBLBTC'
                           , 'DATABTC']

grafico = [] #Será criado um gráfico para facilitar visualização dos lucros em cada mês
linha0 = [] #Linha no gráfico que irá representar 0%
count = 0 #Contador criado para criar o gráfico

#Peso das indicações de compra e venda:

xcci = 0
xrsi = 1
xwt = 0
xuo = 0
xosc = 0
xwr = 0
xmarker = 0
xcmrsi = 1
xbbb = 1
xtsi = 0
xfisher = 0
ximi = 0
xvixfix = 0
xrvi = 0
xsmi = 1
xstoch = 0
xvx3 = 0
xcct = 0
xcmo = 0
xcm = 1 #Excelente indicador
xmacd = 1

for intervalo in lista_intervalo:
    for simbolo in binance_simbolos_USANDO:
        for data_inicio in lista_data:

            if intervalo == '15m':
                mult = 4 #1h = 4 * 15m
            elif intervalo == '30m':
                mult = 2 #1h = 2 * 30m
            elif intervalo == '1h':
                mult = 1 #1h = 1 * 1h
            elif intervalo == '5m':
                mult = 12 #1h = 12 * 5m

            count += 1
            data = data_binance(data_inicio, simbolo, intervalo) #Extraindo a base de dados

            base = data
            base = base.reset_index() #Resetando os índices, para que começem no 0

            #Extraindo os dados da base em forma de colunas:

            Date = date(base)
            High = high(base)
            Low = low(base)
            Open = opeN(base)
            Close = cloSe(base)
            Volume = volume(base)
            Preco = preco(High, Low, Close)

            '''
             Coluna de compra ou venda.
             Caso o valor seja positivo, indica a compra. Negativo a venda e nulo a indecisão
            '''

            Comprar_Vender = pd.DataFrame(np.zeros((len(base), 1))) #Coluna de tamanho len(base) com entradas 0, mas em formato de DataFrame
            Comprar_Vender = Comprar_Vender.squeeze() #Transformando de Dataframe para Series

                                                    #Sinais

            #Abaixo estão as indicações de compra e venda que os indicadores das funções
            #acima fornecem. Favor entrar nos sites nas descrições de cada função para
            #saber mais sobre as recomendações de compra e venda.

            '''
             CCI
            '''

            CCI = cci(20, 0.015, Preco)

            for i in range(1,len(base)):
                if CCI[i - 1] > 200 and CCI[i] < 200:
                    Comprar_Vender[i] += -xcci
                elif CCI[i - 1] < -200 and CCI[i] > -200:
                    Comprar_Vender[i] += xcci

            '''
             RSI
            '''

            RSI = rsi(14, Close)
            
            for i in range(1,len(base)):
                if RSI[i - 1] > 70 and RSI[i] < 70:
                    Comprar_Vender[i] += -xrsi
                elif RSI[i - 1] < 30 and RSI[i] > 30:
                    Comprar_Vender[i] += xrsi

            '''
             WT(WaveTrend Oscillator)
            '''

            WT = wt(10, 21, 0.015, Preco)
                
            for i in range(1,len(base)):
                if WT[i - 1] > 60 and WT[i] < 60:
                    Comprar_Vender[i] += -xwt
                elif WT[i - 1] < -60 and WT[i] > -60:
                    Comprar_Vender[i] += xwt
            
            '''
            UO
            '''
            
            Ult_Osc = uo(7, 14, 28, 4, 2, 1, Low, High, Preco, Close)
          
            for i in range(1,len(base)):
                if Ult_Osc[i - 1] > 70 and Ult_Osc[i] < 70:
                    Comprar_Vender[i] += -xuo
                elif Ult_Osc[i - 1] < 30 and Ult_Osc[i] > 30:
                    Comprar_Vender[i] += xuo
            
            '''
             Stochastic Oscillator
            '''
              
            Osc_Est = osc_est(14, Low, High, Close)
           
            for i in range(1,len(base)):
                if Osc_Est[i - 1] > 80 and Osc_Est[i] < 80:
                    Comprar_Vender[i] += -xosc
                elif Osc_Est[i - 1] < 20 and Osc_Est[i] > 20:
                    Comprar_Vender[i] += xosc
            
            '''
             Williams %R
            '''
            
            William_R = williamr(14, High, Low)
          
            for i in range(1,len(base)):
                if William_R[i - 1] > -20 and William_R[i] < -20:
                    Comprar_Vender[i] += -xwr
                elif William_R[i - 1] < -80 and William_R[i] > -80:
                    Comprar_Vender[i] += xwr
            
            '''
             DeMarker
            '''
            
            De_Marker = demarker(13, High, Low)
           
            for i in range(1,len(base)):
                if De_Marker[i - 1] > 0.7 and De_Marker[i] < 0.7:
                    Comprar_Vender[i] += -xmarker
                elif De_Marker[i - 1] < 0.3 and De_Marker[i] > 0.3:
                    Comprar_Vender[i] += xmarker
            
            '''
             CM_RSI Plus EMA
            '''
            
            CMRSI, EMACRSI = cm_rsi_ema(14, 14, Close) #Original: 20, 10, Close
            
            for i in range(1,len(base)):
                if ((CMRSI[i - 1] > 70 and CMRSI[i] < 70)  #Original: 80
                or (EMACRSI[i - 1] > 70 and EMACRSI[i] < 70)):
                    Comprar_Vender[i] += -xcmrsi
                elif ((CMRSI[i - 1] < 30 and CMRSI[i] > 30)  #Original: 20
                or (EMACRSI[i - 1] < 30 and EMACRSI[i] > 30)):
                    Comprar_Vender[i] += xcmrsi
            
            '''
             BB %B(Bolinger Bands %B)
            '''
            
            BB_B = bb_b(20, 2, Close)
        
            for i in range(1,len(base)):
                if BB_B[i - 1] > 1 and BB_B[i] < 1:
                    Comprar_Vender[i] += -xbbb
                elif BB_B[i - 1] < 0 and BB_B[i] > 0:
                    Comprar_Vender[i] += xbbb
            
            '''
             TSI
            '''
            
            TSI = tsi(25, 13, Close)
            
            for i in range(1,len(base)):
                if TSI[i - 1] > 40 and TSI[i] < 40:
                    Comprar_Vender[i] += -xtsi
                elif TSI[i - 1] < -40 and TSI[i] > -40:
                    Comprar_Vender[i] += xtsi
            
            '''
             Fisher
            '''
            
            Fisher = fisher(High, Low, 9)
            
            for i in range(1,len(base)):
                if Fisher[i - 1] > 1.5 and Fisher[i] < 1.5:
                    Comprar_Vender[i] += -xfisher
                elif Fisher[i - 1] < -1.5 and Fisher[i] > -1.5:
                    Comprar_Vender[i] += xfisher
            
            '''
             IMI(Intraday Momentum Index)
            '''
            
            IMI = imi(14, Close, Open)
           
            for i in range(1,len(base)):
                if IMI[i - 1] > 70 and IMI[i] < 70:
                    Comprar_Vender[i] += -ximi
                elif IMI[i - 1] < 30 and IMI[i] > 30:
                    Comprar_Vender[i] += ximi
            
            
            '''
             CM_Williams_Vix_Fix
            '''
            
            WVF, WVF_linha_superior, WVF_alcancemaior = cm_williams_vix_fix(22, 20, 2, 50, 0.85, 1.01, Close)
            
            for i in range(1,len(base)):
                if WVF[i] >= WVF_linha_superior[i] or WVF[i] >= WVF_alcancemaior[i]:
                    Comprar_Vender[i] += xvixfix
        
            '''
             RVI original e RVI revisão
            '''
            
            RVI_revisao, RVI_original = rvi(10, 10, High, Close)
            
            for i in range(1,len(base)):
                if ((RVI_original[i - 1] > 60 and RVI_original[i] < 60) 
                or (RVI_revisao[i - 1] > 60 and RVI_revisao[i] < 60)):
                    Comprar_Vender[i] += xrvi
                elif ((RVI_original[i - 1] < 40 and RVI_original[i] > 40) 
                or (RVI_revisao[i - 1] < 40 and RVI_revisao[i] > 40)):
                    Comprar_Vender[i] += -xrvi
            
            '''
             SMI Ergodic
            '''
            
            SMIE, SMIESinal = smie(20, 5, 5, Close)
            
            for i in range(1,len(base)):
                if (SMIE[i - 1] > SMIESinal[i - 1] and SMIE[i] < SMIESinal[i] 
                and SMIE[i] > 0.1 and SMIESinal[i - 1] > 0.1):
                    Comprar_Vender[i] += -xsmi
                elif (SMIE[i - 1] < SMIESinal[i - 1] and SMIE[i] > SMIESinal[i]
                and SMIE[i - 1] < -0.1 and SMIESinal[i] < -0.1):
                    Comprar_Vender[i] += xsmi
            
            '''
             StochRSI
            '''
            
            StochRSI, StochRSISinal = stochrsi(14, 3)

            for i in range(1,len(base)):
                if (StochRSI[i - 1] > StochRSISinal[i - 1] and StochRSI[i] < StochRSISinal[i] 
                and StochRSI[i] > 0.1 and StochRSISinal[i - 1] > 0.1):
                    Comprar_Vender[i] += -xstoch
                elif (StochRSI[i - 1] < StochRSISinal[i - 1] and StochRSI[i] > StochRSISinal[i]
                and StochRSI[i - 1] < -0.1 and StochRSISinal[i] < -0.1):
                    Comprar_Vender[i] += xstoch
            
            '''
             Stoch_VX3
            '''
            
            Stoch_VX3 = stoch_vx3(0.5, 0.3, 144, High, Low)
            
            for i in range(1,len(base)):
                if Stoch_VX3[i - 1] > 80 and Stoch_VX3[i] < 80:
                    Comprar_Vender[i] += -xvx3
                elif Stoch_VX3[i - 1] < 20 and Stoch_VX3[i] > 20:
                    Comprar_Vender[i] += xvx3

            '''
             CCT Bollinger Band Oscillator
            '''
            
            CCTBBO, CCTBBOSinal = cctbbo(21, 13, Close)
            
            for i in range(1,len(base)):
                if (CCTBBO[i - 1] > 100 and CCTBBO[i] < 100 
                and CCTBBOSinal[i] > 50 and CCTBBOSinal[i - 1] > 50):
                    Comprar_Vender[i] += -xcct
                elif (CCTBBO[i - 1] < 0 and CCTBBO[i] > 0 
                and CCTBBOSinal[i] < 50 and CCTBBOSinal[i - 1] < 50):
                    Comprar_Vender[i] += xcct
            
            '''
             Chande Momentum Oscillator(CMO)
            '''
                       
            CMO = cmo(9, Close)
        
            for i in range(1,len(base)):
                if CMO[i - 1] > 50 and CMO[i] < 50:
                    Comprar_Vender[i] += -xcmo
                elif CMO[i - 1] < -50 and CMO[i] > -50:
                    Comprar_Vender[i] += xcmo
    
            '''
             CM Stochastic Multi-TimeFrame
            '''
    
            D, K = cm_stock_mtf(3, 3)
    
            for i in range(len(base) - 1):
                if (D[i] >= 80 and K[i] >= 80 and K[i - 1] > D[i - 1] and D[i] < K[i]):
                    Comprar_Vender[i] += -xcm
                elif (D[i] <= 20 and K[i] <= 20 and K[i - 1] < D[i - 1] and D[i] > K[i]):
                    Comprar_Vender[i] += xcm
    
            '''
             MACD
            '''
            
            MACD, MACDSinal = macd(12, 26, 9, Close)
            
            for i in range(1,len(base)):
                if MACD[i - 1] > MACDSinal[i - 1] and MACD[i] < MACDSinal[i]:
                    Comprar_Vender[i] += -xmacd
                elif MACD[i - 1] < MACDSinal[i - 1] and MACD[i] > MACDSinal[i]:
                    Comprar_Vender[i] += xmacd
    
            '''
             Verificando os lucros
            '''
            
            #Verifica o lucro relizado da criptomoeda no determinado período de tempo
            #calculando o lucro entre uma indicação de compra e a proxima indicação de venda
            #e somando todas as porcentagens
            
            lucro = [] #Lista que acumula lucros e prejuízos
            
            #Usamos pedaços a mais no inicio da base pois a precisão de alguns indicadores
            #depende muito da quantidade de dados. O lucro é calculado do dia seguinte da data
            #de início até o dia atual
            
            #Programação abaixo é apenas para gráficos de 15m, 30m e 1h
            

            Comprar_Vender = Comprar_Vender[mult * 24:len(base)] #Excluindo os primeiros dados da base, pois tornam os indicadores imprecisos
            Comprar_Vender = Comprar_Vender.reset_index(drop = True) #Resetando índices, para não gerar conflitos na hora da comparação com Close
            Close = Close[mult * 24:len(base)] #Excluindo os primeiros dados da base, pois tornam os indicadores imprecisos
            Close = Close.reset_index(drop = True) #Resetando índices, para não gerar conflitos na hora da comparação com Comprar_Vender

            i = 0
            j = 0
            quantidade_indicadores_compra = 1
            quantidade_indicadores_venda = -1
            
            while i < len(Comprar_Vender):
                if Comprar_Vender[i] >= quantidade_indicadores_compra:
                    while j < len(Comprar_Vender):
                        if i + j >= len(Comprar_Vender):
                            i = len(Comprar_Vender)
                            break
                        elif Comprar_Vender[j + i] <= quantidade_indicadores_venda:
                            lucro.append((Close[j + i] - Close[i]) / Close[i]) #Porcentagem entre uma indicação de compra e a outra de venda
                            i = j + i
                            break
                        j += 1
                i += 1
                
            print('Lucro de {:.5f}% desde {} até agora'.format(100 * sum(lucro), data_inicio.replace(data_inicio[0:2], '%s'%(int(data_inicio[0:2]) + 1))))
            
            from datetime import datetime #importando novamente datetime de datetime para evitar conflito
            
            #Criando gráfico mostrando o par de moedas, junto ao intervalo, relacionando a data 
            #de início com o somatória total de lucro em cada data de início
            
            grafico.append(sum(lucro)) #Acumulando lucros de cada mês
            linha0.append(0) #Acumulando zeros
            
            if count == len(lista_data):
                count = 0
                plt.plot(grafico[::-1]) #Plotando grafico
                plt.plot(linha0) #Plotando linha0               
                plt.ylabel('{}, {}'.format(simbolo, intervalo)) #Legenda ao lado esquerdo do eixo y
                plt.grid()
                plt.show() #Mostrando gráfico
                
                #Resetando o gráfico:
                
                grafico = []
                linha0 = []                