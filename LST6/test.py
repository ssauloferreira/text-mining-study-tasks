from glob import glob
import pickle

print('loading vocabulary...')
with open('vocabulary', 'rb') as fp:
    vocabulary = pickle.load(fp)
print('vocabulary has been loaded.')
print(vocabulary)
print(len(vocabulary))