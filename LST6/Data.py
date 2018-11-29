import numpy as np

class Data():

    data = np.array(0)
    label = np.array(0)

    def __init__(self):
        data = np.array(0)
        label =np.array(0)

    def set(self, data, label):
        self.data = data
        self.label = label

    def setData(self, data):
        self.data = data

    def setLabel(self,label):
        self.label = label


