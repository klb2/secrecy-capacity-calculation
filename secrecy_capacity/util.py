""" Utility functions.

Copyright (C) 2020-2021 Karl-Ludwig Besser

License:
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Author: Karl-Ludwig Besser, Technische Universitaet Braunschweig
"""

import numpy as np
from scipy import sparse

def H(x):
    return x.conj().T

def is_pos_def(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError as e:
        print(e)
        return False
    #return np.all(np.linalg.eigvalsh(x) > 0)

def is_hermitian(matrix):
    return np.allclose(H(matrix), matrix)

def vec_stack_cols(matrix):
    return np.reshape(matrix, (-1, 1), order="F")

def vech_stack_cols_tril(matrix):
    matrix = np.array(matrix)
    idx = np.triu_indices_from(matrix)
    return np.reshape(matrix.T[idx], (-1, 1))

def inv_vec(vector, shape=None):
    #m = (np.sqrt(8*len(vector)+1)-1)/2
    if shape is None:
        m = int(np.sqrt(len(vector)))
        return np.reshape(vector, (m, m), order="F")
    else:
        return np.reshape(vector, shape, order="F")

def inv_vech(vector):
    m = (np.sqrt(8*len(vector)+1)-1)/2
    m = int(m)
    a = np.zeros((m, m))
    idx = np.triu_indices_from(a)
    a[idx] = vector.flat
    a = a.T
    a[idx] = np.conj(vector.flat)
    return a

def duplication_matrix_fast(n):
    """Duplication matrix. Implementation from:
       https://www.mathworks.com/matlabcentral/answers/473737-efficient-algorithm-for-a-duplication-matrix"""
    m = int(n*(n+1)/2)
    n_sq = n**2
    r = 0
    a = 1
    v = np.zeros(n_sq)
    for i in range(n):
        v[r:r+i] = i + 1 - n + np.cumsum(n - np.arange(0, i))
        #print("Indices 1: {}\t{}".format(r, r+i))
        #print(v)
        r = r + i

        #print("Indices 2: {}\t{}".format(r, r+n-i))
        v[r:r+n-i] = np.arange(a, a+n-i)
        r = r + n - i
        a = a + n - i
    return sparse.csr_matrix((np.ones(n_sq), (range(n_sq), v-1)), shape=(n_sq, m))
