import numpy as np
import matrix
import pdb

class NEARZERO():
    def __init__(self):
        value = 1e-10
    def __lt__(self,other):
        return self.value < other
    def __gt__(self,other):
        return self.value > other
    def __le__(self,other):
        return self.value <= other
    def __ge__(self,other):
        return self.value >= other
    def __eq__(self,other):
        return self.value == other
    def __ne__(self,other):
        return self.value != other

def stdev_zero(data,axis):
    std = np.std(data,axis=axis)
    result = np.zeros(std.shape,dtype=bool)
    for i in range(std.shape[0]):
        if std[i]==0:
            result[i] = True
    return result

def read_wine():
    from csv import reader
    matrix = []
    f = open('wine.data', 'r')
    csvReader = reader(f)
    for row in csvReader:
        matrix.append(row)
    f.close()
    matrix = np.array(matrix, dtype=float)
    ### Return (y, X) ###
    return matrix.T[0], np.delete(matrix, 0, 1)

def read_wine_quality(colors=['red']):
    from csv import reader
    matrix = []
    for color in colors:
        f = open('winequality-'+color+'.csv', 'r')
        csvReader = reader(f, delimiter=';')
        next(csvReader, None)
        for row in csvReader:
            matrix.append(row)
        f.close()
    matrix = np.array(matrix)
    matrix = np.array(matrix, dtype=float)
    ### Return (y, X) ###
    return matrix.T[-1], np.delete(matrix, -1, 1)




