from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile, GenericUnivariateSelect, \
    SelectFromModel
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from spacy.compat import pickle
import LST6.DocToVec as dtv

print('loading vocabulary...')
with open('vocabulary', 'rb') as fp:
    vocabulary = pickle.load(fp)
print('vocabulary has been loaded.')

# PROCESSING THE WORDS
dictionary = dtv.toProcess(vocabulary,500)

# GETTING THE VECTORS
documents = dtv.generateBOW(dictionary)

# FEATURE SELECTION
clf = ExtraTreesClassifier(n_estimators=10)
clf = clf.fit(documents.data, documents.label)
model = SelectFromModel(clf, prefit=True)
result = model.transform(documents.data)
print(result.shape)

# CLUSTERING
clustering = SpectralClustering(n_clusters=3,
        random_state=0).fit_predict(result)

# CALCULATING THE METRICS
    # F-MEASURE
f_measure = f1_score(documents.label, clustering, average='weighted')
print(f_measure)
    # ACCURACY
accuracy = accuracy_score(documents.label, clustering)
print(accuracy)
    # CONFUSION MATRIX
confMatrix = confusion_matrix(documents.label, clustering)
print(confMatrix)