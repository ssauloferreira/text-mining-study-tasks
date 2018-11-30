from glob import glob
import pickle
import numpy as np


a = [[1,2,3], [3,5,6]]
b = np.array([[1,2,3]])

np.insert(b,a,axis=1)

print(b)