import os
import nltk
import math

def probabilidade(unigrams, frase):
    frase = frase.lower()
    tokens_frase = tokenizar_simples(frase)
    proba = []

    V = len(list(unigrams.values()))
    N = sum(list(unigrams.values()))

    for token in tokens_frase:
        if token in unigrams.keys():
            aux1 = unigrams[token] + 1
        else:
            aux1 = 1

        aux2 = N + V
        proba.append(aux1 / aux2)

    soma = 0
    for p in proba:
        soma += math.log2(p)

    return soma


def unigram(tokens):
    aux = {}

    for sentence in tokens:
        for token in sentence:
            if token not in aux.keys():
                aux[token] = 1
            else:
                cont = aux[token]
                aux[token] = cont + 1

    unigram = {}
    N = sum(list(aux.values()))

    for token in aux.keys():
        unigram[token] = aux[token]/N

    return unigram

def palavras_minuscula(sentences_total):
    sentences_total = [sentence.lower() for sentence in sentences_total]
    return sentences_total

def sentence_splitting(texto):
    sent_detector = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    sentences = sent_detector.tokenize(texto.strip())
    return sentences

def sentence_splitting_total(sentences, sentences_total):
    for sentence in sentences: sentences_total.append(sentence)
    return sentences_total

def tokenizar(sentences_total):
    tokens = [nltk.word_tokenize(sentence, language='portuguese') for sentence in sentences_total]
    return tokens

def tokenizar_simples(sentence):
    return nltk.word_tokenize(sentence, language='portuguese')

def ler_arquivos(locais_arquivos):
    sentences_total = []
    for local in locais_arquivos:
        arquivo = open(local, 'r')
        texto = arquivo.read()

        # Sentenca por arquivo
        sentences = sentence_splitting(texto)
        # Sentencas do corpus
        sentences_total = sentence_splitting_total(sentences, sentences_total)

        arquivo.close()

    return sentences_total

def treinamento():
    # Pegar todos os arquivos existentes na pasta:
    local_main = 'Noticias_Portugues\\'
    locais_arquivos = [os.path.join(local_main, nome) for nome in os.listdir(local_main)]
    # Vetor com todas as sentences do corpus
    sentences_total = ler_arquivos(locais_arquivos)
    # Tornar tudo minusculo
    sentences_total = palavras_minuscula(sentences_total)
    # Tokenizar as frases
    tokens = tokenizar(sentences_total)
    # Gerar unigrams
    return unigram(tokens)

def main():
    unigrams = treinamento()

    frase = input("Informe a frase desejada: ")

    resposta = probabilidade(unigrams, frase)

    print('Probabilidade dessa frase ocorrer é de %.2f.' %resposta)


if __name__ == "__main__":
    main()
