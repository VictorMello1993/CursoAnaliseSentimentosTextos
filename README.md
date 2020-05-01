# CursoAnaliseSentimentosTextos
Repositório contendo scripts desenvolvidos ao longo do curso de análise de sentimentos em textos, utilizando a biblioteca NLTK e algoritmo de Naive Bayes.

Recursos:
* NLTK
* Pandas
* NumPy
* codecs
* matplotlib (em breve serão implementados os gráficos de desempenho do algoritmo e da matriz de confusão)

Os scripts foram desenvolvidos no Visual Studio Code. Para testá-los é preciso ficar atento com o encoding do arquivo txt, o qual foi utilizado para preencher a base de dados. Deixar sempre no formato UTF-8, para que o interpretador consiga reconhecer os acentos, ç, entre outros ao chamar o método print().

O próximo passo é incluir a diretiva ```# -*- coding: utf-8 -*-``` no início do arquivo do código-fonte, pois posteriormente será utilizado codecs para ler o arquivo txt conforme o encoding configurado no passo anterior.

ATENÇÃO: sempre chamar close() caso o arquivo não seja mais utilizado para assim economizar memória.

No Google Colab não será necessário importar a biblioteca codecs nem incluir a diretiva, pois o servidor já possui interpretador previamente configurado para ler qualquer enconding do arquivo automaticamente, basta utilizar ```open('BaseTreinamento.txt', encoding='utf-8')```. Novamente, não esquecer de deixar o encoding configurado no arquivo conforme citado no primeiro passo.

Dicas de como conseguir mais stopwords: https://gist.github.com/alopes/5358189

OBS: É importante ressaltar que os resultados do treinamento não foram os melhores, pois foi utilizado simplesmente para estudar o funcionamento geral da biblioteca. Caso deseje utilizar a mesma base de dados, é importante verificar como as frases estão distribuídas na base de treinamento, se estão classificadas corretamente e com mais precisão. Como é aprendizado de máquina supervisionado, muitas vezes se faz necessário recorrer a ajuda de um especialista em linguística para rotular bem as frases com seus respectivos sentimentos para obter resultados melhores. Ou então, procurar outras técnicas que sejam mais precisas. É muito difícil conseguir bases de dados de frases em português que sejam confiáveis para fins de estudos científicos ou para trabalhar em situações reais (redes sociais como Twitter, por exemplo, já ajudam nessa questão). Muitas empresas provêm soluções utilizando machine learning construindo as suas próprias bases de dados.

Também é fundamental verificar as probabilidades em função do número de classes (emoções). Geralmente, quanto menor (mínimo 2 classes), melhor.

Verificar também se existe algum caracter ou termo inadequado que foi utilizado por engano no treinamento do algoritmo, como existência de mais stopwords.
