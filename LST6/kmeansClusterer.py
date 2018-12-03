import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, GenericUnivariateSelect, SelectPercentile, \
    RFECV, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.svm import LinearSVC
from spacy.compat import pickle
import LST6.DocToVec as dtv

print('loading vocabulary...')
with open('vocabulary', 'rb') as fp:
    vocabulary = pickle.load(fp)
print('vocabulary has been loaded.')

# PROCESSING THE WORDS
dictionary = dtv.toProcess(vocabulary,500)

# GETTING THE VECTORES
documents = dtv.generateTFIDF(dictionary)
data = np.array(documents.data)
labels = np.array(documents.label)

# FEATURE SELECTION
clf = ExtraTreesClassifier(n_estimators=10)
clf = clf.fit(documents.data, documents.label)
model = SelectFromModel(clf, prefit=True)
result = model.transform(documents.data)
print(result.shape)

# CLUSTERING
kmeans = KMeans(n_clusters=3)
KModel = kmeans.fit(result)

# CALCULATING THE METRICS
    # F-MEASURE
f_measure = f1_score(documents.label, KModel.labels_, average='weighted')
print(f_measure)
    # ACCURACY
accuracy = accuracy_score(documents.label, KModel.labels_)
print(accuracy)
    # CONFUSION MATRIX
confMatrix = confusion_matrix(documents.label, KModel.labels_)
print(confMatrix)