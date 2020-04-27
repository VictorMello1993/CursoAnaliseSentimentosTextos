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
