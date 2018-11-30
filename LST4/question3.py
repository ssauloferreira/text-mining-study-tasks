# -*- coding: utf-8 -*-
import nltk

def realizar_troca_aux (n_grams_mensagem, n_grams_corpus):
    trocas = {}

    cont = 0
    list_aux = list(n_grams_corpus.items())
    for i in n_grams_mensagem.items():
        trocas[i[0]] = list_aux[cont][0]
        cont += 1

    return trocas

def realizar_troca(mensagem, bigrams_corpus, unigrams_corpus):
    unigram_mensagem, bigram_mensagem = preparar_mensagem(mensagem)

    trocas_unigrans = realizar_troca_aux(unigram_mensagem, unigrams_corpus)
    trocas_bigrams = realizar_troca_aux(bigram_mensagem, bigrams_corpus)

    mensagem = concatenar_mensagem(mensagem)

    print('[INFO] Substituicao unigram: ')
    saida = ""
    for i in mensagem:
        saida += trocas_unigrans[i]

    print(saida)

    print('[INFO] Substituicao bigram: ')
    saida = ""
    for i in range(len(mensagem[:-1])):
        bigram_aux = mensagem[i] + mensagem[i+1]
        saida += trocas_bigrams[bigram_aux]

    print(saida)

def concatenar_mensagem(mensagem):
    mensagem = mensagem.split(" ")
    corpus_mensagem = ""
    for i in mensagem: corpus_mensagem += i

    return corpus_mensagem

def preparar_mensagem(mensagem):
    corpus_mensagem = concatenar_mensagem(mensagem)

    unigram_mensagem = unigram(corpus_mensagem)
    unigram_mensagem = ordenar(unigram_mensagem)
    bigram_mensagem = bigram(corpus_mensagem, unigram_mensagem)
    bigram_mensagem = ordenar(bigram_mensagem)

    return unigram_mensagem, bigram_mensagem

def ordenar(n_gram):
    cont = 0
    n_gram_principais = {}
    for item in sorted(n_gram, key=n_gram.get, reverse=True):
        n_gram_principais[item] = n_gram[item]
        cont += 1

    return n_gram_principais

def n_grams(corpus):
    unigrams = unigram(corpus)
    unigrams_principais = ordenar(unigrams)

    bigrams = bigram(corpus, unigrams)
    bigrams_principais = ordenar(bigrams)

    return bigrams_principais, unigrams_principais

def ocorrencia_par_palavras (string, corpus):
    ocorrencia = 0

    if type(corpus) == list:
        for sentence in corpus:
            ocorrencia += sentence.count(string)
    elif type(corpus) == str:
        ocorrencia = corpus.count(string)

    return ocorrencia

def bigram(corpus, unigrams):
    tokens = tokenizar_corpus(corpus)
    bigrams_aux = list(nltk.bigrams(tokens))

    bigrams = {}

    for item in bigrams_aux:
        string = item[0] + item[1]
        ocorrencia_wi_wj = ocorrencia_par_palavras(string, corpus)
        ocorrencia_wj = unigrams[item[0]]

        bigrams[string] = ocorrencia_wi_wj/ocorrencia_wj

    return bigrams

def unigram(corpus):
    freq_token = frequenia_tokens(corpus)

    N = sum(list(freq_token.values()))
    unigrams = {}

    for itens in freq_token.keys():
        unigrams[itens] = freq_token[itens]/N

    return unigrams

def frequenia_tokens(corpus):
    tokens = tokenizar_corpus(corpus)

    freq_token = {}

    for i in tokens:
        if i not in freq_token.keys():
            freq_token[i] = 1
        else:
            cont = freq_token[i]
            freq_token[i] = cont + 1

    return freq_token

def tokenizar_corpus(corpus):
    tokens = []
    for pos_sentence in range(len(corpus)):
        sentence = corpus[pos_sentence]
        sentence = sentence.lower()
        for pos_caracter in range(len(sentence)):
            if sentence[pos_caracter].isalnum():
                tokens.append(sentence[pos_caracter])

    return tokens

def ler_arquivo():
    corpus = []

    with open("corpus.txt", encoding='utf-8') as file:
        for line in file:
            line_split = line.split("\t")
            sentence = line_split[1]
            corpus.append(sentence)

    return corpus

def main():
    ler_arquivo()

    corpus = ler_arquivo()

    bigrams, unigrams = n_grams(corpus)

    mensagem = input("Informe a mensagem: ")
    mensagem = mensagem.lower()

    realizar_troca(mensagem, bigrams, unigrams)







if __name__ == "__main__":
    main()