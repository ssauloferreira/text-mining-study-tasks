import nltk
import numpy as np

insCost = 1
subsCost = 2

def toProcess(text):
    text = nltk.word_tokenize(text)
    text = [word.lower() for word in text if word.isalpha()]

    return text

def p(wordA, wordB, i, j):
    if wordA[i] == wordB[j]:
        return 0
    else:
        return subsCost


def getDistance(wordA, wordB):
    wordA = " " + wordA
    wordB = " " + wordB
    lenA = wordA.__len__()
    lenB = wordB.__len__()
    matrix = np.zeros((lenA, lenB), dtype=np.int)

    for i in range (0, lenA):
        for j in range (0, lenB):
            if i == 0:
                matrix[i][j] = j
            elif j == 0:
                matrix[i][j] = i
            else:
                custos = []
                custos.append(matrix[i-1][j]+insCost)
                custos.append(matrix[i][j-1]+insCost)
                custos.append(matrix[i-1][j-1]+p(wordA,wordB, i, j))
                matrix[i][j] = min(custos)
        if i%2 == 0:
            print(matrix)
            print()

    return matrix[lenA-1][lenB-1]


def getSimilar(word, wordlistFile, distance):
    arq = open(wordlistFile, 'r', errors='ignore')
    wordlist = arq.read()
    tokens = toProcess(wordlist)

    similars = []

    for token in tokens:
        dis = getDistance(word, token)
        if dis <= distance and token not in similars:
            similars.append(token)
    return similars

def verifyExistence(word, wordlistFile):
    arq = open(wordlistFile, 'r', errors='ignore')
    wordlist = arq.read()

    word = '\n'+word+'\n'
    if word in wordlist:
        return 1
    else:
        return 0