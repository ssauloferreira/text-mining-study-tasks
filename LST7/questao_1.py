'''
Links:
http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
https://textminingonline.com/getting-started-with-word2vec-and-glove-in-python
https://radimrehurek.com/gensim/models/word2vec.html
https://rare-technologies.com/word2vec-tutorial/
https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
http://www.scikit-yb.org/en/latest/api/text/tsne.html#
'''

import gensim
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
from yellowbrick.text import TSNEVisualizer

vocabulario = [['banana'], ['potato'], ['pear'], ['pineapple'], ['apple'], ['turtle'], ['peacock'], ['dog'], ['cat'],
               ['duck'], ['swan'], ['elephant'], ['pig'], ['lion'], ['penguin'], ['cup'], ['bowl'], ['kettle'],
               ['spoon'], ['car'], ['truck'], ['ship'], ['helicopter'], ['boat'], ['pen'], ['pencil'], ['knife'],
               ['scissors'], ['screwdriver']]

print('[INFO] Carregando o modelo...')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print('1.1 - Resposta :')

pares_palavras = []
similaridades = []

print('[INFO] Iniciando calculo de similaridade...')
for word1 in vocabulario:
    word1 = word1[0]
    for word2 in vocabulario:
        word2 = word2[0]
        if word1 != word2:
            similaridade = model.wv.similarity(w1=word1, w2=word2)
            pares_palavras.append(word1 + '-' + word2)
            similaridades.append(similaridade)
            print("Similaridade %s e %s : %.3f" %(word1, word2, similaridade))

    print("\n")

print('---------------------------------------------------------------------------')
print('1.3 - Resposta:')

model = Word2Vec(vocabulario, min_count=1)

X = model[model.wv.vocab]

for NUM_CLUSTERS in range(2, 6):
    print('NUM_CLUSTERS %d' %(NUM_CLUSTERS))
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        print (word + ":" + str(assigned_clusters[i]))

    print('\n')

    tsne = TSNEVisualizer()
    tsne.fit(X, assigned_clusters)
    tsne.poof(outpath="k" + str(NUM_CLUSTERS) +".png")

print('---------------------------------------------------------------------------')
print('1.4 - Resposta:')
