from glob import glob
from sklearn.cluster import KMeans
import nltk
import numpy as np
from nltk import WordNetLemmatizer
from spacy.compat import pickle
from LST6.Data import Data

lemmatizer = WordNetLemmatizer()

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

    data = np.array(data)
    labels = np.array(labels)

    documents = Data()
    documents.setData(data)
    documents.setLabel(labels)

    return documents


def main():
    print('loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        vocabulary = pickle.load(fp)
    print('vocabulary has been loaded.')

    dictionary = toProcess(vocabulary,2500)
    print(dictionary)
    documents = generateBOW(dictionary)

    for doc in documents.data:
        print(doc)

    kmeans = KMeans(n_clusters=3)
    KModel = kmeans.fit(documents.data)
    print(KModel.labels_)


if __name__ == '__main__':
    main()