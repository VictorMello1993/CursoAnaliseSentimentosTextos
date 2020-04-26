# -*- coding: utf-8 -*-
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.metrics import ConfusionMatrix
import codecs

# Obtendo um array de tuplas de frases com seus respectivos sentimentos (classes) preenchendo cada linha de um arquivo txt


def obterBaseDados(arquivo):
    base = []
    for linha in arquivo:
        linha = linha.rstrip()
        frase = linha[:linha.rfind(',')+1]
        emocao = linha[linha.rfind(',')+1:]
        frases = tuple([frase, emocao])
        base.append(frases)
    return base


'''OBS1: foi separado 30% da base de treinamento para avaliação.
    De 160 frases que eram de alegria, 48 serão avaliados. E as outras 120 frases para cada uma das outras cinco classes, 36 serão avaliados. 
    Portanto, das frases que são de alegria, 112 serão utilizados para treinamento, e o restante serão utilizados 124 frases para treinamento.
    OBS2: Os registros da base de treinamento não devem constar na base de testes'''

stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']

# Lista todos os stopwords da língua portuguesa da NLTK
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')

# Algumas palavras não constam na lista de stopwords da NLTK. Para isso, basta acrescentar mais palavras utilizando append ou extend
stopwordsnltk.extend(['vou', 'tão', 'vai', 'mim', 'por',
                      'ser', 'assim', 'com', 'melhor'])


arquivo_treinamento = codecs.open('BaseTreinamento.txt', 'r', 'utf-8')
arquivo_teste = codecs.open('BaseTeste.txt', 'r', 'utf-8')

content_treinamento = arquivo_treinamento.readlines()
content_teste = arquivo_teste.readlines()

arquivo_treinamento.close()
arquivo_teste.close()

base_treinamento = obterBaseDados(content_treinamento)
base_teste = obterBaseDados(content_teste)

# ---------------------------------------------------------PRÉ-PROCESSAMENTO----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Retorna uma lista de tuplas contendo tokens de frases e a classe (emoção) com recurso de eliminação de stopwords
def removeStopWords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

# Aplica stemming de palavras presentes numa base de dados com stopwords removidos
def stemming1(texto):
    stemmer = nltk.stem.RSLPStemmer()  # Stemmizador utilizado na língua portuguesa
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p))
                       for p in palavras.split() if p not in stopwordsnltk]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

# Aplica stemming de palavras presentes numa base de dados sem remover stopwords, para evitar que o algoritmo de Naive Bayes não encontre correspondência na tabela de probabilidades
def stemming2(texto):
    frasesstemming = []
    stemmer = nltk.stem.RSLPStemmer()  # Stemmizador utilizado na língua portuguesa
    for palavras in texto.split():
        comstemming = [str(stemmer.stem(p)) for p in palavras.split()]
        frasesstemming.extend(comstemming)
    return frasesstemming


frases_stemming_treinamento = stemming1(base_treinamento)
frases_stemming_teste = stemming1(base_teste)
# print(frases_stemming_teste)

# Obtendo todas as palavras de uma lista de stemming
def buscaPalavras(frases):
    todasPalavras = []
    for(palavras, emocao) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras


palavras_treinamento = buscaPalavras(frases_stemming_treinamento)
palavras_teste = buscaPalavras(frases_stemming_teste)
# print(palavras_treinamento)


def buscaFrequenciaPalavras(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras


frequencia_treinamento = buscaFrequenciaPalavras(palavras_treinamento)
frequencia_teste = buscaFrequenciaPalavras(palavras_teste)
# print(frequencia_treinamento.most_common(50)) #Obtendo a frequência das primeiras 50 palavras

# Buscando pela chave da tupla obtida na função BuscaFrequencia que é única para cada registro
def buscaPalavrasUnicas(frequencia):    
    freq = frequencia.keys()
    return freq

palavras_unicas_treinamento = buscaPalavrasUnicas(frequencia_treinamento)
palavras_unicas_teste = buscaPalavrasUnicas(frequencia_teste)

# Extraindo as palavras presentes em uma determinada frase - retorna um dicionário de características onde a chave corresponde uma palavra, e o valor representa um booleano onde será setado como True se a palavra está presente numa frase ou False se não está presente.
def extratorPalavras(documento):
    doc = set(documento)
    dict_caracteristicas = {}
    for palavras in palavras_unicas_treinamento:
        dict_caracteristicas['%s' % palavras] = (palavras in doc)
    return dict_caracteristicas


caracteristicasFrases = extratorPalavras(['am', 'nov', 'dia'])
# print(caracteristicasFrases)

# Obtendo as bases de dados completa (treinamento e testes)
base_completa_treinamento = nltk.classify.apply_features(extratorPalavras, frases_stemming_treinamento)
base_completa_teste = nltk.classify.apply_features(extratorPalavras, frases_stemming_teste)

# print(np.asarray(base_completa_treinamento).shape) #Verificando o tamanho da base
tamanho_base_treinamento = sum(len(i) for i in base_completa_treinamento)
tamanho_base_teste = sum(len(i) for i in base_completa_teste)

# Base de dados redimensionada para melhor visualização
# base_reshape_treinamento = np.reshape(base_completa_treinamento, tamanho_base_treinamento)
# base_reshape_teste = np.reshape(base_completa_treinamento, tamanho_base_teste)
# print(base_reshape_treinamento)
# print(base_reshape_teste)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------Treinamento do modelo utilizando Naive Bayes-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)  # Construindo a tabela de probabilidades
print(classificador.labels())  # Obtendo as classes

# print(classificador.show_most_informative_features(5))
'''Obtendo 5 características nos quais os atributos (palavras) são os mais informativos. Por exemplo, se a palavra 'dia' estiver presente em uma frase, significa que a probabilidade de uma
frase ser de alegria é 2.3 vezes maior do que de medo. Porém, esses valores são menores porque a base de dados é menor, tem poucos registros.'''

# teste = 'estou com medo'
teste = 'Game of Thrones vai ser do caralho!'

testestemming = stemming2(teste)
# print(testestemming)

# Obtendo um registro de características da frase stemmizada
novo = extratorPalavras(testestemming)
# print(novo)

# Efetivamente está realizando a classificação da frase em questão (somente label). O método classify() é responsável pelos cálculos de estimativas de probabilidades para cada classe e verificar a maior delas
print(classificador.classify(novo))

# Verificando as probabilidades de cada classe - Retorna label junto com as probabilidades
distribuicao = classificador.prob_classify(novo)
# for classe in distribuicao.samples():
#     print(f'{classe}: {distribuicao.prob(classe)}')

'''OBS: O stemming da NLTK não considera o contexto da frase, e sim palavra por palavra. Por esse motivo, existem frases, como 'eu te amo', em que 
    a palavra 'amo' não foi extraída para obter o seu radical, o que acabou influenciando no resultado final dos cálculos de probabilidades.
    Em vez de interpretar a frase como alegria, acabou interpretando como medo. A alternativa para melhorar os resultados é buscar um 
    outro stemmizador que trabalhe com mais precisão com essas palavras
'''

# ------------------------------------------------------------------Avaliação do algoritmo--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Acurácia na base de teste - Foi obtido 34% para alegria
print(nltk.classify.accuracy(classificador, base_completa_teste))

'''3 formas de avaliação a verificar:
    1. Cenário (possivelmente as frases rotuladas na base de dados tiveram a classificação incorreta, provavelmente não foram rotuladas com auxílio de um especialista em linguística)
    2. Número de classes (para 6 classes temos que o acerto mínimo seria em torno de 16%)
    3. Algoritmo Zero Rules (classificar novos registros para uma classe que possui maior número de registros) - No caso da alegria, o acerto mínimo seria de 21,05%
'''

erros = []
for (frase, classe) in base_completa_teste:
    resultado = classificador.classify(frase)
    if resultado != classe:  # Se o resultado da classificação for diferente da classe prevista na base de testes, significa que houve um erro e será acrescentado no array de erros
        erros.append((classe, resultado, frase))

# for (classe, resultado, frase) in erros: #Mostrando todas as frases que tiveram erros na classificação
#     print(classe, resultado, frase)

# Obtendo a matriz de confusão
esperado = []
previsto = []
for(frase, classe) in base_completa_teste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)
matriz = ConfusionMatrix(esperado, previsto)
print(matriz)
