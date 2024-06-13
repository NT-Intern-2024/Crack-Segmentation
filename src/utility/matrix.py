import numpy as np

def print_matrix(np_array: np, info: str = ""):
    print(info)
    # print(np.asmatrix(np_array), end="\n\n")
    print(np.array2string(np_array, threshold= np.inf), end="\n\n")