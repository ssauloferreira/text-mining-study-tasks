#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec


local = 'rt-polaritydata'
stop_words = set(stopwords.words('english'))

def fun_remove_stop_words(texto):
    # tokenização:
    words = word_tokenize(texto)
    words_filter = []

    # remove stop words:
    for w in words:
        if w not in stop_words:
            words_filter.append(w)

    return words_filter


def fun_remove_simb_especiais(words):
    filter_words = []
    for w in words:
        if re.match("^[a-zA-Z0-9_]*$", w):
            filter_words.append(w)

    return filter_words


def fun_sum_vector_words(model, words, i):
    vector = None
    for w in words:
        vector = vector + model.wv[w]




def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # i = 0 neg
    # i = 1 pos
    for i, arquivo in enumerate(os.listdir(local)):
        arquivo = open(local + '/' + arquivo)
        texto = arquivo.read()

        texto = texto.lower()

        sentences = sent_tokenize(texto)

        for sentence in sentences:
            words = fun_remove_stop_words(sentence)
            words = fun_remove_simb_especiais(words)

            fun_sum_vector_words(model, words, i)

        break


if __name__ == "__main__":
    main()
