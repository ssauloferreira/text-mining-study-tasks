# Tokenization
# Sentence Splitting
# Lemmatization
# Stemming
# POS tagging

import nltk
import matplotlib.pyplot as plt

arq = open("texto_en.txt", "r")

texto = arq.read()

# ----- Tokenization -------
tokens = nltk.word_tokenize(texto)
print("[INFO] Tokenization:")
print(tokens)

# ----- Sentence Splitting -----
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sent_detector.tokenize(texto.strip())
print("[INFO] Sentence Splitting:")
print(sentences)

# ----- Lemmatization (Reduce inflections or variant forms to base form)------
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lema = {}
for w in tokens:
    if w not in lema:
        lema[w] = wordnet_lemmatizer.lemmatize(w)
print("[INFO] Lematization:")
print(lema)

# ----- Stemming (Reduce terms to their stems or roots)-----
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
stem = {}
for w in tokens:
    if w != ',' and w != '.' and w != '!' and w != '?' and w != '"' and w != "''" and w != '``':
        if w not in stem:
            stem[w] = porter_stemmer.stem(w)
print("[INFO] Stemming:")
print(stem)

# ----- POS tagging -----
tagged = nltk.pos_tag(tokens)
print("[INFO] POS tagging:")
print(tagged)

arq.close()

print("\n")
split = []
for w in tokens:
    if w != ',' and w != '.' and w != '!' and w != '?' and w != '"' and w != "''" and w != '``':
        split.append(w)
print("Letra A: ", len(split))  # Resposta 693
print("Letra B: ", len(stem.keys()))   # Resposta 326

tokens_per_sentence = []
for s in sentences:
    token = nltk.word_tokenize(s)
    tokens_per_sentence.append(len(token))

print("Letra C: {0} sentencas e {1:2.4f} media de tokens por sentenca".format(len(sentences), (sum(tokens_per_sentence)/len(tokens_per_sentence))))

print("Letra E: Grafico em anexo")

tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)

freq = []
tags = []
for i in tag_fd.most_common():
    tags.append(i[0])
    freq.append(i[1])

# ------- Grafico questão 08 letra e ---------
plt.figure(0)
plt.bar(tags, freq)
plt.xlabel("TAGS")
plt.ylabel("Frequency")
plt.title("Question 8 Letter E")
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
plt.savefig('letter_e.png')

print("Letra F: ")
stem_cont = {}
for w in tokens:
    if w != ',' and w != '.' and w != '!' and w != '?' and w != '"' and w != "''" and w != '``':
        w_aux = porter_stemmer.stem(w)
        if w_aux not in stem_cont:
            stem_cont[w_aux] = 1
        elif w_aux in stem_cont:
            cont = stem_cont[w_aux]
            stem_cont[w_aux] = cont + 1

from operator import itemgetter
stem_cont = dict(reversed(sorted(stem_cont.items(), key=itemgetter(1))))

freq = list(stem_cont.values())
stems = list(stem_cont.keys())
print(freq)
print(stems)

# ------- Grafico questão 08 letra e ---------
plt.figure(1)
plt.bar(stems[0:15], freq[0:15])
plt.xlabel("Steamming")
plt.ylabel("Frequency")
#plt.yticks(sorted(set(list(stem_cont.values()))))
plt.title("Question 8 Letter F")
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
plt.savefig('letter_f.png')