import nltk

texto1 = 'Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow'
texto2 = 'Olá, meu nome é Victor, tudo bem?'
# print(texto1.split('.')) #Maneira manual (versão em Python)

# frases = nltk.tokenize.sent_tokenize(texto1) #Separação de frases usando o recurso da NLTK (a maneira mais eficiente do que em forma manual)
frases = nltk.tokenize.sent_tokenize(texto2, language='portuguese') #Separação de frases usando o recurso da NLTK (a maneira mais eficiente do que em forma manual)
print(frases)

# tokens = nltk.word_tokenize(texto1) #Tokenização de frases
tokens = nltk.word_tokenize(texto2) #Tokenização de frases
print(tokens)

classes = nltk.pos_tag(tokens) #Obtendo as classes gramaticais de palavras numa frase
print(classes)

#Mais informações sobre as classes de palavras se encontram na documentação https://cs.nyu.edu/grishman/jet/guide/PennPOS.html

entidades = nltk.chunk.ne_chunk(classes) #Obtendo entidades de palavras (se é uma pessoa, empresa, etc.)
print(entidades)


