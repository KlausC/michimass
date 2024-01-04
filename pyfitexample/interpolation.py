#! /usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from numpy.testing import assert_allclose


def d(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def d2(a, A):
    return np.array(list(d(a, b) for b in A))

def weight(d, dd: list, p):
    s = np.min(dd)
    a = 0.1
    # return (d + a/ s + a) ** p
    return np.exp((s - d) / (s + a) * (-p))


def t(z, A, p):
    return sum(x**p for x in d2(z, A))

def fmi(M, z, A, p):
    n = len(A)
    if n != len(M):
        raise ValueError("Dimension Mismatch")
    
    return sum(M[i] * d(z, A[i]) ** p / t(z, A, p) for i in range(len(M)))


def fkl(M, z, A: np.array, p):
    n = len(A)
    if n != len(M):
        raise ValueError("Dimension Mismatch")
    
    x, y = z[0], z[1]
    H = np.array([np.ones(n), A[:, 0], A[:, 1]]).T
    dd = d2(z, A)
    ww = list(weight(d, dd, p / 2) for d in dd)
    D = np.diag(ww)
    DH = np.matmul(D, H)
    B = np.matmul(linalg.pinv(DH), np.matmul(D, M))
    return np.dot(B.T, [1, x, y])


def interpol(z, f, A: list, M: list, p: int = -3):
    return f(M, z, A, p)

def make_pkl(A, M, p):
    return lambda x, y: interpol([x, y], fkl, A, M, p)

def make_pmi(A, M, p):
    return lambda x, y: interpol([x, y], fmi, A, M, p)


def plot(f, x, y, xnew, ynew):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), height_ratios = [1.0])
    znew = np.array([[f(x, y) for x in xnew] for y in ynew])

    im = plt.contourf(xnew, ynew, znew, levels=np.linspace(-0.1, 1.1, num=25))
    ax1.set_aspect('equal')
    fig.colorbar(im, ax = ax1)
    ax1.plot(x, y, "co", markersize = 1)

    return znew

def example(make, p, x, y, M):
    xx = np.linspace(min(x) - 0.1, max(x) + 0.1, num=100)
    yy = np.linspace(min(y) - 0.1, max(y) + 0.1, num=100)
    A = np.array([list(t) for t in zip(x, y)])
    pf = make(A, M, p)
    return plot(pf, x, y, xx, yy)

x = [1.0, 2, 3, 1, 2, 3]
y = [1.0, 1, 1, 2, 2, 2]
M = [0.0, 0, 0, 0, 1, 1]

example(make_pkl, -2, x, y, M)
example(make_pmi, -3, x, y, M)
plt.show()
