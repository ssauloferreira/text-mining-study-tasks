import os

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from stanfordcorenlp import StanfordCoreNLP


STOP_WORDS = set(stopwords.words('english'))
nlp = StanfordCoreNLP("C:\stanford-corenlp-full-2018-02-27")

def sentence_spliting(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_detector.tokenize(text)


def remove_stop_words(tokens):
    words_filter = []

    # remove stop words:
    for w in tokens:
        if w not in STOP_WORDS and w.isalnum():
            words_filter.append(w)

    return words_filter


def tokenizar(sentences):
    tokens_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tokens = remove_stop_words(tokens)
        tokens_sentences.append(tokens)

    return tokens_sentences


def calcular_pesos_feature1(dicio):
    soma = 0
    for tf_idf in dicio.values():
        soma += tf_idf

    return soma


'''
http://www.ultravioletanalytics.com/blog/tf-idf-basics-with-pandas-scikit-learn
https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments
'''


def feature_1(tokens_sentences):  # TF-IDF dos termos de uma sentenca
    peso_sentencas = []
    vectorizer = TfidfVectorizer(strip_accents=ascii, stop_words='english', analyzer='word', min_df=1)

    for sentence in tokens_sentences:
        train_fit = vectorizer.fit(sentence)

        idf = vectorizer.idf_
        # armazena o tf-idf referente aos termos de uma frase:
        dicio = dict(zip(train_fit.get_feature_names(), idf))

        # armazena o peso total referente a frase:
        peso_sentencas.append(calcular_pesos_feature1(dicio))

    return peso_sentencas


def update_dicio_words(dependency_parse):
    peso = 0
    for dep in dependency_parse:
        if 'subj' in dep[0]:
            peso += 3
        elif 'obj' in dep[0]:
            peso += 2
        else:
            peso += 1

    return peso


def convert_dicio(tokens):
    dicio = {}
    for w in tokens:
        if w not in dicio:
            dicio[w] = 0

    return dicio


'''
Atribuir os pesos de acordo com a função gramática da palavra na frase:
* Peso = 3,  se o termo é um sujeito
* Peso = 2,  se o termo é um objeto
* Peso = 1,  todos os outros casos
'''
def feature_2(sentences, tokens_sentences):
    peso_sentencas = []

    for i, sentence in enumerate(sentences):
        tokens = tokens_sentences[i]
        dependency_parse = nlp.dependency_parse(sentence)
        peso_sentence = update_dicio_words(dependency_parse)

        peso_sentencas.append(peso_sentence)

    return peso_sentencas


def feature_3(peso_sentencas_feature_2, sentences):
    peso_sentencas = []
    max_tamanho = 0

    for sentence in sentences:
        if len(sentence) > max_tamanho:
            max_tamanho = len(sentence)

    for sentence in peso_sentencas_feature_2:
        peso = sentence/max_tamanho
        peso_sentencas.append(peso)

    return peso_sentencas


def main():
    local_global = 'News_LE_09/'

    news = os.listdir(local_global)

    tf_df_sentences = []

    for n in news:
        local = local_global + n + '/'
        arq = os.listdir(local)

        f = open(local + arq[1], 'r')

        text = f.read()

        # lower case:
        text = text.lower()

        sentences = sentence_spliting(text)
        tokens_sentences = tokenizar(sentences)

        # feature 1
        # retorna uma matriz, onde cada posição contem o peso total da frase
        pesos_frases_feature_1 = feature_1(tokens_sentences)
        print(pesos_frases_feature_1)
        # feature 2
        pesos_frases_feature_2 = feature_2(sentences, tokens_sentences)
        print(pesos_frases_feature_2)
        # feature 3
        print(pesos_frases_feature_3)
        pesos_frases_feature_3 = feature_3(pesos_frases_feature_2, sentences)


if __name__ == "__main__":
    main()
