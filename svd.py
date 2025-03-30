#!/usr/bin/env python

import math
import numpy as np
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm, PowerNorm

def bidiag(A, img):  # assumes m >= n
    m, n = A.shape
    if m < n:
        return A

    def householder_matrix(y):
        a = math.copysign(1, y[0]) * np.linalg.norm(y, 2)
        sigma = (y[0] + a) / a
        y[0] += a; r = y / y[0]
        return np.identity(y.shape[0]) - sigma * np.outer(r, r)

    # iterate over columns
    count = n if n != m else n - 1
    for i in range(count):
        # eliminate over the column
        y = A[i:, i].copy()
        Q = householder_matrix(y)
        A[i:, i:] = Q @ A[i:, i:]

        img.set_data(np.absolute(A))
        plt.draw()
        plt.pause(0.001)

        if i + 1 < n:
            # eliminate over the row
            y = A[i, i+1:].copy()
            P = householder_matrix(y)
            A[i:, i+1:] = A[i:, i+1:] @ P

            img.set_data(np.absolute(A))
            plt.draw()
            plt.pause(0.001)

    return A

def givensrotation(a, b):
    if b != 0.0:
        hyp = np.sqrt(a**2 + b**2)
        d = 1.0 / hyp
        cos = abs(a) * d
        sin = math.copysign(d, a) * b
        r = math.copysign(1.0, a) * hyp
    else:
        cos = 1.0
        sin = 0.0
        r = a
    return cos, sin, r

def wilkinson_shift(B):
    # https://web.stanford.edu/class/cme335/lecture5
    # where B is bidiagonal and T is tridiagonal
    m, n = B.shape
    a = np.dot(B[:, n-1], B[:, n-1]) # T[n-1, n-1]
    b = np.dot(B[:, n-2], B[:, n-1]) # T[n-1, n-2]
    c = np.dot(B[:, n-2], B[:, n-2]) # T[n-2, n-2]

    d = (c - a) / 2.0
    return a + d - math.copysign(1.0, d) * np.sqrt(d**2 + b**2)

def diag(B, img):
    m, n = B.shape

    R = np.identity(n)  # right (column)
    L = np.identity(m)  # left (row)

    converged = False
    while not converged:
        # introduce a bulge
        # shift = B[n-1, n-1] # Wilkinson's shift (I don't think this is true or correct)
        # form the requisite elements of T = B.T @ B
        a = np.dot(B[:, 0], B[:, 0]) # T[0, 0]
        b = np.dot(B[:, 1], B[:, 0]) # T[1, 0]
        #shift = np.dot(B[:, n-1], B[:, n-1]) # T[m-1, m-1]
        shift = wilkinson_shift(B)

        # apply a givens rotation G to the right (columns) to introduce a bulge below the main diagonal
        cos, sin, _ = givensrotation(a - shift, b)
        B[:, 0], B[:, 1] = (B[:, 0] * cos) + (B[:, 1] * sin), (B[:, 1] * cos) - (B[:, 0] * sin)
        img.set_data(np.absolute(B))
        plt.draw()
        plt.pause(0.001)

        # apply a new givens rotation G.T-hat to the left (rows) to zero out the nonzero placed below the main diagonal
        cos, sin, _ = givensrotation(B[0, 0], B[1, 0])
        B[0], B[1] = (B[0] * cos) + (B[1] * sin), (B[1] * cos) - (B[0] * sin)
        img.set_data(np.absolute(B))
        plt.draw()
        plt.pause(0.001)

        # chase out the bulge
        for i in range(1, n-1):

            # apply a givens rotation G to the right (columns)
            cos, sin, _ = givensrotation(B[i-1, i], B[i-1, i+1])
            B[:, i], B[:, i+1] = (B[:, i] * cos) + (B[:, i+1] * sin), (B[:, i+1] * cos) - (B[:, i] * sin)
            img.set_data(np.absolute(B))
            plt.draw()
            plt.pause(0.001)

            # apply a new givens G.T-hat rotation to the left (rows)
            cos, sin, _ = givensrotation(B[i, i], B[i+1, i])
            B[i], B[i+1] = (B[i] * cos) + (B[i+1] * sin), (B[i+1] * cos) - (B[i] * sin)
            img.set_data(np.absolute(B))
            plt.draw()
            plt.pause(0.001)

        # check for convergence
        # https://web.stanford.edu/class/cme335/lecture5
        # they check the subdiagonal here
        #if np.dot(B[:, n-2], B[:, n-1]) < 1e-16:
        #if B[n-2, n-1] < 1e-16: # the last super diagonal element
        if B[0, 1] < 1e-16: # the first super diagonal element
            converged = True

    return B

def ordersv(A):
    # reorder singular values in descending fashion
    pass

def collect_singular_values(A):
    count = min(A.shape)
    singular_values = []

    for i in range(count):
        singular_values.append(A[i, i])

    return singular_values

if __name__ == "__main__":
    A = np.random.rand(10, 10)
    C = np.array([
        [4, 1, 3],
        [2, 3, 1],
        [1, 1, 2],
    ], dtype="float")
    #m, n = C.shape
    #print(C[1, 0])
    #print(C[n-2, n-2])
    #print(C[:, 1])
    #plt.ion()

    fig, ax = plt.subplots()
    img = ax.imshow(np.absolute(A))
    A = bidiag(A, img)

    B = diag(A, img)
    img.set_data(np.absolute(B))

    plt.ioff()
    plt.show()
