import nltk
import pickle
from glob import glob

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
vocabulary = {}
documents = []
i = 1

for filepath in glob('News_CNN_3_classes_30\\**'):
    for file in glob(filepath + '\\**'):
        arq = open(file, 'r')
        text = arq.read()

        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        documents.append(text)

        for token in text:
            if token in vocabulary:
                j = vocabulary[token]
                j += 1
                vocabulary[token] = j
            elif token not in vocabulary:
                vocabulary[token] = 1

        i += 1

print('saving vocabulary')
with open('vocabulary', 'wb') as fp:
    pickle.dump(vocabulary, fp)

idf = {}

for word in vocabulary.keys():
    count = 0
    for text in documents:
        if word in text:
            count += 1
    idf[word] = count

print(idf)
print('saving idfs')
with open('idf', 'wb') as fp:
    pickle.dump(idf, fp)