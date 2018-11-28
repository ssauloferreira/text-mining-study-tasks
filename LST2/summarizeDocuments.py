import nltk

def pipeline(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(text.strip())
    qtt = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        postag = nltk.tag.pos_tag(tokens)
        parse_tree = nltk.ne_chunk(nltk.tag.pos_tag(tokens), binary=True)  # POS tagging before chunking!

        named_entities = []

        soma = []

        for t in parse_tree.subtrees():
            if t.label() == 'NE':
                named_entities.append(list(t))  # if you want to save a list of tagged words instead of a tree
                soma.append(list(t).__len__())


        named_entities.clear()
        qtt.append(sum(soma))
    return sentences, qtt


def getScore(sentences,qtt):
    scoreSi = []
    i = 0
    n = qtt.__len__()
    while i < qtt.__len__():
        score = 1- ((i+1)/n)
        scoreGlobal = 1 + ((2*qtt[i])/(n+score))
        scoreSi.append([sentences[i], scoreGlobal])
        i = i+1
    return scoreSi


def summarized(score):
    score.sort(key=lambda x: x[1], reverse = True)
    n = score.__len__()
    k = 0.3*n
    k = int(float(k))
    for i in range(0,n-k):
        score.pop()

    summarize = ""
    for sentence in score:
        summarize = summarize + sentence[0]

    return summarize