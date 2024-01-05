#! /usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from scipy.interpolate import LinearNDInterpolator


def d(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def d2(a, A):
    return np.array(list(d(a, b) for b in A))


def weight(d, dd: list, p):
    s = np.min(dd)
    a = 0.1
    # return (d + a/ s + a) ** p
    return np.exp((s - d) / (s + a) * p)


def t(z, A, p):
    return sum(x ** (-p) for x in d2(z, A))


def fmi(M, z, A, p):
    n = len(A)
    if n != len(M):
        raise ValueError("Dimension Mismatch")

    return sum(M[i] * d(z, A[i]) ** (-p) / t(z, A, p) for i in range(len(M)))


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


# The signed area of the triangle given by *a*, *b*, *c*
def area(a, b, c):
    A = [[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]]
    return linalg.det(A)


def phi(x, p):
    return x if p != 0 else type(x)(1) 

"""
[Barycentric Interpolation
4 Barycentric interpolation over polygons](see: https://www.inf.usi.ch/faculty/hormann/papers/Hormann.2014.BI.pdf)
"""
def fba(F, x, xx, p):
    n = len(xx)

    def XX(i):
        return xx[i if i < n else i - n]

    A = [area(x, xx[i], XX(i + 1)) for i in range(n)]
    B = [area(x, xx[i - 1], XX(i + 1)) for i in range(n)]
    R = [phi(d(x, xx[i]), p) for i in range(n)]

    def RR(i):
        return R[i if i < n else i - n]

    def WW(i):
        aa = np.prod([(A[k] if np.mod(i - k, n) > 1 else 1.0) for k in range(n)])
        return (RR(i + 1) * A[i - 1] - R[i] * B[i] + R[i - 1] * A[i]) * aa

    W = [WW(i) for i in range(n)]

    W = np.divide(W, sum(W))

    return sum(W[i] * F[i] for i in range(n))


def interpol(z, f, A: list, M: list, p: int = -3):
    return f(M, z, A, p)


def make_pkl(A, M, p):
    return lambda x, y: interpol([x, y], fkl, A, M, p)


def make_pmi(A, M, p):
    return lambda x, y: interpol([x, y], fmi, A, M, p)


def make_pli(A, M, p):
    return LinearNDInterpolator(A, M, fill_value=0.0)


def make_pba(A, M, p):
    n = len(A)
    if n != len(M):
        raise ValueError("Dimension Mismatch")

    return lambda x, y: interpol([x, y], fba, A, M, p)


def plot(f, x, y, xnew, ynew):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), height_ratios=[1.0])
    znew = np.array([[f(x, y) for x in xnew] for y in ynew])

    im = plt.contourf(xnew, ynew, znew, levels=np.linspace(-0.1, 1.2, num=27))
    ax1.set_aspect("equal")
    fig.colorbar(im, ax=ax1)
    ax1.plot(x, y, "co", markersize=1)

    return znew


def example(make, p, x, y, M):
    xx = np.linspace(min(x) - 0.1, max(x) + 0.1, num=100)
    yy = np.linspace(min(y) - 0.1, max(y) + 0.1, num=100)
    A = np.array(list(zip(x, y)))
    pf = make(A, M, p)
    return plot(pf, x, y, xx, yy)


x = [1.0, 2, 3, 3, 2, 1]
y = [1.0, 1, 1, 2, 1.2, 2]
M = [0.0, 0, 0, 1, 1, 0]

#example(make_pkl, 2, x, y, M)
#example(make_pmi, 3, x, y, M)
#example(make_pli, 0, x, y, M)
example(make_pba, 1, x, y, M)
example(make_pba, 0, x, y, M)

# fba(M, [1.9, 1.9], list(zip(x, y)), 0)

plt.show()
