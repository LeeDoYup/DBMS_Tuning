import matrix
import numpy as np
def is_numeric_matrix(matrix):
    return not "S" in matrix.dtype.str

def is_lexical_matrix(matrix):
    return "S" in matrix.dtype.str
