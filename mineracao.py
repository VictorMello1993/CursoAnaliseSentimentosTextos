import nltk
import numpy as np
import pandas as pd

base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]

# print(base[0]) #Primeira frase

stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']

stopwordsnltk = nltk.corpus.stopwords.words('portuguese') #Lista todos os stopwords na língua portuguesa utilizando recurso da NLTK - Mais eficiente do que usar uma lista personalizada conforme declarada na linha anterior
# print(stopwordsnltk)

#---------------------------------------------------------PRÉ-PROCESSAMENTO----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Retorna uma lista de tuplas contendo tokens de frases e a classe (emoção) com recurso de eliminação de stopwords
def removeStopWords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

# print(removeStopWords(base))

#Aplica stemming de palavras presentes numa base de dados com stopwords removidos
def stemming1(texto):
    stemmer = nltk.stem.RSLPStemmer() #Stemmizador utilizado na língua portuguesa
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]        
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

#Aplica stemming de palavras presentes numa base de dados sem remover stopwords, para evitar que o algoritmo de Naive Bayes não encontre correspondência na tabela de probabilidades
def stemming2(texto):
    frasesstemming = []
    stemmer = nltk.stem.RSLPStemmer() #Stemmizador utilizado na língua portuguesa
    for palavras in texto.split():        
        comstemming = [str(stemmer.stem(p)) for p in palavras.split()]        
        frasesstemming.extend(comstemming)
    return frasesstemming

frasesstemming = stemming1(base)
# print(frasesstemming)

#Obtendo todas as palavras de uma lista de stemming
def buscaPalavras(frases):
    todasPalavras = []
    for(palavras, emocao) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras

palavras = buscaPalavras(frasesstemming)
# print(palavras)

def buscaFrequenciaPalavras(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscaFrequenciaPalavras(palavras)
# print(frequencia.most_common(50)) #Obtendo a frequência das primeiras 50 palavras

def buscaPalavrasUnicas(frequencia):
    freq = frequencia.keys() #Buscando pela chave da tupla obtida na função BuscaFrequencia que é única para cada registro
    return freq

palavrasUnicas = buscaPalavrasUnicas(frequencia)

#Extraindo as palavras presentes em uma determinada frase - retorna um dicionário de características onde a chave corresponde uma palavra, e o valor representa um booleano onde será setado como True se a palavra está presente numa frase ou False se não está presente.
def extratorPalavras(documento):
    doc = set(documento)
    dict_caracteristicas = {}
    for palavras in palavrasUnicas:
        dict_caracteristicas['%s' % palavras] = (palavras in doc)
    return dict_caracteristicas

caracteristicasFrases  = extratorPalavras(['am', 'nov', 'dia'])
# print(caracteristicasFrases)

#Obtendo uma base de dados completa
baseCompleta = nltk.classify.apply_features(extratorPalavras, frasesstemming)
# print(baseCompleta)

# print(np.asarray(baseCompleta).shape) #Verificando o tamanho da base (neste caso, a base possui 20 linhas com 2 colunas, totalizando 40 elementos)
tamanho_base = sum(len(i) for i in baseCompleta)

baseReshape = np.reshape(baseCompleta, tamanho_base) #Base de dados redimensionada para melhor visualização
# print(baseReshape)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------Treinamento do modelo utilizando Naive Bayes-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

classificador = nltk.NaiveBayesClassifier.train(baseCompleta) #Construindo a tabela de probabilidades
# print(classificador.labels()) #Obtendo as classes 

# print(classificador.show_most_informative_features(5)) 
'''Obtendo 5 características nos quais os atributos (palavras) são os mais representativos. Por exemplo, se o dia estiver presente em uma frase, significa que a probabilidade de uma
frase ser de alegria é 2.3 vezes maior do que de medo. Porém, esses valores são menores porque a base de dados é menor, tem poucos registros.'''

# teste = 'estou com medo'
teste = 'amor medo dia apavorado'
testestemming = stemming2(teste)
print(testestemming)

novo = extratorPalavras(testestemming) #Obtendo um registro de características da frase stemmizada 'est com med'
# print(novo)

#Efetivamente está realizando a classificação da frase em questão (label). O método classify() é responsável pelos cálculos de estimativas de probabilidades para cada classe e verificar a maior delas
print(classificador.classify(novo))

#Verificando as probabilidades de cada classe - Retorna label junto com as probabilidades
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
    print(f'{classe}: {distribuicao.prob(classe)} %')


'''OBS: O Stemming da NLTK não considera o contexto da frase, e sim palavra por palavra. Por esse motivo, existem frases, como 'eu te amo', em que 
a palavra 'amo' não foi extraída para obter o seu radical, o que acabou influenciando no resultado final dos cálculos de probabilidades.
Em vez de interpretar a frase como alegria, acabou interpretando como medo. A alternativa para melhorar os resultados é buscar um 
outro stemmizador que trabalhe com mais precisão com essas palavras.'''