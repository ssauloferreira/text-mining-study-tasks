from glob import glob
import pickle
import LST6.DocToVec as dtv
import nltk

def frequence(dictionary):
    i = 0
    wf = {}
    documents = []
    for filepath in glob('News_CNN_3_classes_30\\**'):
        a = []
        for file in glob(filepath + '\\**'):
            ## Reading the document and processing it
            arq = open(file, 'r')
            text = arq.read()
            text = nltk.word_tokenize(text)
            text = [word.lower() for word in text if word.isalpha()]

            a.append(a)
        i = i+1

        for x in wf:
            print(x)
            print(wf[x])

def main():
    print('loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        vocabulary = pickle.load(fp)
    print('vocabulary has been loaded.')

    dictionary = dtv.toProcess(vocabulary, 1000)

    frequence(dictionary)


if __name__ == '__main__':
    main()