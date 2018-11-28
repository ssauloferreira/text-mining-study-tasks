import matplotlib.pyplot as plt
from nltk.corpus import brown

news_text = brown.words(categories='news')

words = [word.lower() for word in news_text if word.isalpha()]

vocabulary = []
qtd = []
for word in words:
    if word not in vocabulary:
        vocabulary.append(word)
        qtd.append(1)
    else:
        for i in range(0, vocabulary.__len__()):
            if vocabulary[i] == word:
                qtd[i] += 1


frequency = []

for i in range(0, vocabulary.__len__()):
    frequency.append([vocabulary[i], qtd[i]])

frequency.sort(key=lambda x: x[1], reverse = True)

vocabulary = []
qtd = []
print(frequency[:50])
frequency = frequency[:15]

for i in frequency:
    vocabulary.append(i[0])
    qtd.append(i[1])


plt.bar(vocabulary,qtd)

plt.savefig('image.png')
