import nltk

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

print(base[0]) #Primeira frase

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

print(removeStopWords(base))

#Aplica stemming de palavras presentes numa base de dados com stopwords removidos
def stemming(texto):
    stemmer = nltk.stem.RSLPStemmer() #Stemmizador utilizado na língua portuguesa
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]        
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasesstemming = stemming(base)
print(frasesstemming)

#Obtendo todas as palavras de uma lista de stemming
def buscaPalavras(frases):
    todasPalavras = []
    for(palavras, emocao) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras

palavras = buscaPalavras(frasesstemming)
print(palavras)

def buscaFrequenciaPalavras(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscaFrequenciaPalavras(palavras)
print(frequencia.most_common(50)) #Obtendo a frequência das primeiras 50 palavras

def buscaPalavrasUnicas(frequencia):
    freq = frequencia.keys() #Buscando pela chave da tupla obtida na função BuscaFrequencia que é única para cada registro
    return freq

palavrasUnicas = buscaPalavrasUnicas(frequencia)
print(palavrasUnicas)

#Extraindo as palavras presentes em uma determinada frase - retorna um dicionário de características onde a chave corresponde uma palavra, e o valor representa um booleano onde será setado como True se a palavra está presente numa frase ou False se não está presente.
def extratorPalavras(documento):
    doc = set(documento)
    dict_caracteristicas = {}
    for palavras in palavrasUnicas:
        dict_caracteristicas['%s' % palavras] = {palavras in doc}
    return dict_caracteristicas

caracteristicasFrases  = extratorPalavras(['am', 'nov', 'dia'])
print(caracteristicasFrases)

#Obtendo uma base de dados completa
baseCompleta = nltk.classify.apply_features(extratorPalavras, frasesstemming)
print(baseCompleta)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------Treinamento do modelo utilizando Naive Bayes-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------