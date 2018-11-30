import math
from glob import glob

import numpy as np
from sklearn.cluster import KMeans
import nltk
from nltk import WordNetLemmatizer
from nltk.cluster import KMeansClusterer
from sklearn.metrics import confusion_matrix
from spacy.compat import pickle
from LST6.Data import Data

lemmatizer = WordNetLemmatizer()
with open('idf', 'rb') as fp:
    idf = pickle.load(fp)

def getTF(text, word):
    count = 0

    for token in text:
        if word == token:
            count = count+1

    return count/text.__len__()

def getIDF(word):
    if idf[word] == 0:
        return 0
    return math.log2(90/idf[word])

def toProcess(vocabulary,n):

    print('processing vocabulary')

    ## Reading stop-words
    arq = open('stopwords.txt', 'r')
    stopWords = arq.read()
    stopWords = nltk.word_tokenize(stopWords)

    ## Generating a vector of 2-dimensional vectors that will
    ## bin the word and its frequence

    filteredVocabulary = []
    for w in vocabulary:
        if w not in stopWords:
            filteredVocabulary.append([w,vocabulary[w]])

    ## Sorting by frequence and getting the n-firsts words
    ## from vocabulary
    filteredVocabulary.sort(key=lambda x: x[1], reverse=True)
    dictionary = filteredVocabulary[0:n]

    arq.close()

    return dictionary

def generateTFIDF(dictionary):
    i = 0
    data = []
    labels = []

    for filepath in glob('News_CNN_3_classes_30\\**'):
        for file in glob(filepath + '\\**'):
            ## Reading the document and processing it
            arq = open(file, 'r')
            text = arq.read()
            text = nltk.word_tokenize(text)
            text = [word.lower() for word in text if word.isalpha()]

            textLine = []

            for word in dictionary:
                if word[0] in text:
                    textLine.append(getTF(text, word[0])*getIDF(word[0]))
                else:
                    textLine.append(0)

            data.append(textLine)
            labels.append(i)

        i = i+1

    documents = Data()
    documents.setData(data)
    documents.setLabel(labels)

    return documents

def generateBOW(dictionary):
    i = 0
    data = []
    labels = []

    for filepath in glob('News_CNN_3_classes_30\\**'):
        for file in glob(filepath + '\\**'):

            ## Reading the document and processing it
            arq = open(file, 'r')
            text = arq.read()
            text = nltk.word_tokenize(text)
            text = [word.lower() for word in text if word.isalpha()]


            textLine = []

            for word in dictionary:
                if word[0] in text:
                    textLine.append(1)
                else:
                    textLine.append(0)

            data.append(textLine)
            labels.append(i)

        i = i+1

    documents = Data()
    documents.setData(data)
    documents.setLabel(labels)

    return documents

def main():
    print('loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        vocabulary = pickle.load(fp)
    print('vocabulary has been loaded.')

    dictionary = toProcess(vocabulary,500)
    print(dictionary)

    documents = generateBOW(dictionary)

    vectors = [np.array(f) for f in documents.data]

    clusterer = KMeansClusterer(3, nltk.cluster.cosine_distance, repeats=100, avoid_empty_clusters=True)
    clusters = clusterer.cluster(vectors, True)
    confMatrix = confusion_matrix(documents.label, clusters)

    print(confMatrix)




if __name__ == '__main__':
    main()