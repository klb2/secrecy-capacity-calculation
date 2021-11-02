""" Utility functions for Loyka algorithm.

Copyright (C) 2020 Karl-Ludwig Besser

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





# The following functions are taken from: Taken from:
# https://github.com/statsmodels/statsmodels/pull/4143/files
def duplication_matrix(n):
    """Duplication matrix. Taken from: https://github.com/statsmodels/statsmodels/pull/4143/files"""
    l = L(n)
    ltl = l.T.dot(l)
    k = K(n)
    d = l.T + k.dot(l.T) - ltl.dot(k).dot(l.T)
    return d

def vec(x):
    """ravel matrix in fortran order (stacking columns)
    """
    return np.ravel(x, order='F')

def E(i, j, nr, nc):
    """create unit matrix with 1 in (i,j)th element and zero otherwise
    """
    x = np.zeros((nr, nc), np.int64)
    x[i, j] = 1
    return x


def K(n):
    """selection matrix
    symmetric case only
    """
    k = sum(np.kron(E(i, j, n, n), E(i, j, n, n).T)
            for i in range(n) for j in range(n))
    return k

def Ms(n):
    k = K(n)
    return (np.eye(*k.shape) + k) / 2.

def u(i, n):
    """unit vector
    """
    u_ = np.zeros(n, np.int64)
    u_[i] = 1
    return u_


def L(n):
    """elimination matrix
    symmetric case
    """
    # they use 1-based indexing
    # k = sum(u(int(round((j - 1)*n + i - 0.5* j*(j - 1) -1)), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
    k = sum(u(int(np.trunc((j)*n + i - 0.5* (j + 1)*(j))), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
            for i in range(n) for j in range(i+1))
    return k


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    dup_mat = duplication_matrix_fast(args.n)
    print(dup_mat.toarray())
    #dup_mat = duplication_matrix(args.n)
    #dup_mat_sparse = sparse.csc_matrix(dup_mat)
    #sparse.save_npz("duplication_matrix-{}".format(args.n), dup_mat_sparse)
