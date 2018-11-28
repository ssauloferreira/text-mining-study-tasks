import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

def getWCfromFile(fileName, maxWords, pos):
    arq = open(fileName, 'r')
    text = arq.read()
    arq.close()

    lemmatizer = WordNetLemmatizer()
    # tokenizing
    tokens = nltk.word_tokenize(text)
    print(tokens)
    # lemmatizing
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    print(tokens)
    # pos-tagging
    pos_tags = nltk.pos_tag(tokens)
    sw = open('stopwords.txt', 'r')
    stopwords = sw.read()

    words = ""

    if pos == 'noun':
        for i in pos_tags:
            if i[1] == 'NN' or i[1] == 'NNP' or i[1] == 'NNS' or i[1] == 'NNPS':
                if i[0] not in stopwords:
                    words = words + " " + i[0]
    else:
        for i in pos_tags:
            if i[1] == 'VB' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'VBN' or i[1] == 'VBP' or i[1] == 'VBZ':
                if i[0] not in stopwords:
                    words = words + " " + i[0]

    wc = WordCloud(max_words=maxWords).generate(words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()