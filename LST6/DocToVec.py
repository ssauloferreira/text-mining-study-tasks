import math
from glob import glob

import numpy as np
import xlrd
from sklearn.cluster import KMeans
import nltk
from nltk import WordNetLemmatizer
from nltk.cluster import KMeansClusterer
from sklearn.metrics import confusion_matrix
from spacy.compat import pickle
from xlutils.copy import copy

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

def toProcess(vocabulary, n):

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
            text = [lemmatizer.lemmatize(word) for word in text]
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
            text = [lemmatizer.lemmatize(word) for word in text]
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

def printToSheet(dictionary, documents):
    line = []
    line.append(" ")
    for word in dictionary:
        line.append(word)

    lineToSheet(line)
    i = 0
    for doc in documents.data:
        i = i+1

        line = []
        line.append("doc %d" % i)
        for index in doc:
            line.append(index)
        lineToSheet(line)

def lineToSheet(line):
    workbook = xlrd.open_workbook("bow.xls")
    worksheet = workbook.sheet_by_index(0)

    wb = copy(workbook)

    linha = worksheet.nrows
    sheet = wb.get_sheet(0)

    for col in range(len(line)):
        sheet.write(linha, col, str(line[col]))

    try:
        wb.save("bow.xls")
    except IOError:
        wb.save("bow.xls")