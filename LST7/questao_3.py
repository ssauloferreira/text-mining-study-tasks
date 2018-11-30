'''
Links:
https://textminingonline.com/getting-started-with-word2vec-and-glove-in-python
'''

import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def calcular_distancia(word1, lista, dist_max):
    aux = []
    for word2 in lista:
        dist = model.wv.similarity(w1=word1, w2=word2)
        if dist > dist_max:
            if len(aux) > 0:
                aux = []

            aux.append(word1)
            aux.append(word2)

    return aux


def atualizar_dict(dicio, maximos):
    for element in maximos:
        cont = dicio[element[0]]
        dicio[element[0]] = cont + 1

        cont = dicio[element[1]]
        dicio[element[1]] = cont + 1

    return dicio


def buscar_resposta(dicio):
    resposta = ''
    for word in dicio:
        if dicio[word] == 5:
            resposta = word
            break

    return resposta


print('[INFO] Carregando o modelo...')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print('------------------------------------------------------------------------------------------------------')
print('4 - Resposta:')
print('4.1 - Resposta a: ', model.most_similar(positive=["king", "woman"], negative=["man"], topn=5))
print('4.1 - Resposta b: ', model.most_similar(positive=["paris", "italy"], negative=["france"], topn=5))
print('4.1 - Resposta c: Nao foi possivel realizar esse teste, pois na base nao possui a palavra Recife')
print('4.1 - Resposta d: Nao foi possivel realizar esse teste, pois na base nao possui a palavra apple tree')
print('4.1 - Resposta e: ', model.most_similar(positive=["lion", "kangaroo"], negative=["africa"], topn=5))

print('\n')

resposta = ''

vocabulario_a = {'evaluation': 0, 'assessment': 0, 'examination': 0, 'supervision': 0, 'verification': 0}
lista = vocabulario_a.keys()
maximos = []

for i, word1 in enumerate(vocabulario_a):
    dist_max = 0
    if i < (len(lista)-1):
        aux = calcular_distancia(word1, lista[i+1:], dist_max)
    maximos.append(aux)

vocabulario_a = atualizar_dict(vocabulario_a, maximos)

print('4.2 - Resposta a: ' + buscar_resposta(vocabulario_a))

vocabulario_b = {'context': 0, 'meaning': 0, 'significance': 0, 'perspective': 0, 'emphasis': 0}
lista = vocabulario_b.keys()
maximos = []

for i, word1 in enumerate(vocabulario_b):
    dist_max = 0
    if i < (len(lista)-1):
        aux = calcular_distancia(word1, lista[i+1:], dist_max)
    maximos.append(aux)

vocabulario_b = atualizar_dict(vocabulario_b, maximos)

print('4.2 - Resposta b: ' + buscar_resposta(vocabulario_b))

'''vocabulario_c = {'method': 0, 'procedure': 0, 'techinique': 0, 'approach': 0, 'model': 0}
lista = vocabulario_c.keys()
maximos = []

for i, word1 in enumerate(vocabulario_c):
    dist_max = 0
    if i < (len(lista)-1):
        aux = []
        for word2 in lista[i+1:]:
            dist = model.wv.similarity(w1=word1, w2=word2)
            if dist > dist_max:
                if len(aux) > 0:
                    aux = []

                aux.append(word1)
                aux.append(word2)
    maximos.append(aux)


for element in maximos:
    cont = vocabulario_c[element[0]]
    vocabulario_c[element[0]] = cont + 1

    cont = vocabulario_c[element[1]]
    vocabulario_c[element[1]] = cont + 1'''

print('4.2 - Resposta c: Nao eh possivel responder essa questao pois a palavra techinique nao esta no vocabulario utilizado.')

vocabulario_d = {'result': 0, 'outcome': 0, 'effect': 0, 'evidence': 0, 'reason': 0}
lista = vocabulario_d.keys()
maximos = []

for i, word1 in enumerate(vocabulario_d):
    dist_max = 0
    if i < (len(lista)-1):
        aux = calcular_distancia(word1, lista[i+1:], dist_max)
    maximos.append(aux)

vocabulario_d = atualizar_dict(vocabulario_d, maximos)

print('4.2 - Resposta b: ' + buscar_resposta(vocabulario_d))

vocabulario_e = {'conclusion': 0, 'outcome': 0, 'finding': 0, 'assertion': 0, 'explanation': 0}
lista = vocabulario_e.keys()
maximos = []

for i, word1 in enumerate(vocabulario_e):
    dist_max = 0
    if i < (len(lista)-1):
        aux = calcular_distancia(word1, lista[i+1:], dist_max)
    maximos.append(aux)

vocabulario_e = atualizar_dict(vocabulario_e, maximos)

print('4.2 - Resposta b: ' + buscar_resposta(vocabulario_e))
