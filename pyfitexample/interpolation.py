#! /usr/bin/env python3

import warnings
import os

warnings.filterwarnings("ignore")
os.environ["XDG_SESSION_TYPE"] = "xcb"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # für 3d-plots

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

    D = [d(x, a) for a in xx]
    e = 1e-8
    while min(D) < e:
        D = [d(x, a) for a in xx]
        y = np.subtract(np.mean(D, axis=0), x)
        x = np.add(x, np.multiply(y, 2 * e / linalg.norm(y)))
        D = [d(x, a) for a in xx]

    A = [area(x, xx[i], XX(i + 1)) for i in range(n)]
    B = [area(x, xx[i - 1], XX(i + 1)) for i in range(n)]
    R = [phi(d, p) for d in D]

    def RR(i):
        return R[i if i < n else i - n]

    def WW(i):
        aa = np.prod([(A[k] if np.mod(i - k, n) > 1 else 1.0) for k in range(n)])
        ww = (RR(i + 1) * A[i - 1] - R[i] * B[i] + R[i - 1] * A[i]) * aa
        return ww - 1e-100 if ww < 0 else ww + 1e-100

    W = [WW(i) for i in range(n)]

    W = np.divide(W, sum(W))

    return sum(W[i] * F[i] for i in range(n))


def interpol(z, f, A: list, M: list, p: int = -3):
    return f(M, z, A, p)


def verify_M(n: int, M):
    if type(M) == int:
        i = M
        M = np.zeros(n)
        M[np.mod(i, n)] = 1
    else:
        if n != len(M):
            raise ValueError("Dimension Mismatch")
    return M


def make_pkl(A, M, p):
    M = verify_M(len(A), M)
    return lambda x, y: interpol([x, y], fkl, A, M, p)


def make_pmi(A, M, p):
    M = verify_M(len(A), M)
    return lambda x, y: interpol([x, y], fmi, A, M, p)


def make_pli(A, M, p):
    M = verify_M(len(A), M)
    return LinearNDInterpolator(A, M, fill_value=0.0)


def make_pba(A, M, p):
    M = verify_M(len(A), M)
    return lambda x, y: interpol([x, y], fba, A, M, p)


def plot(f, x, y, xnew, ynew, X=None, Y=None):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), height_ratios=[1.0])
    znew = np.array([[f(x, y) for x in xnew] for y in ynew])

    a = round(np.min(znew) - 0.05, ndigits=1)
    b = round(np.max(znew) + 0.05, ndigits=1)
    num = round((b - a) / 0.05) + 1

    levels = np.linspace(a, b, num=num)

    im = plt.contourf(xnew, ynew, znew, levels=levels)
    ax1.set_aspect("equal")
    fig.colorbar(im, ax=ax1)
    ax1.plot(x, y, "co", markersize=1, color="orange")
    if X != None and Y != None:
        ax1.plot(X, Y, "co", markersize=1, color="red")

    x2 = np.outer(xnew, np.ones(len(ynew)))
    y2 = np.outer(np.ones(len(xnew)), ynew)

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 6), height_ratios=[1.0])
    ax2 = plt.axes(projection="3d")
    cmap = plt.get_cmap()
    ax2.contour(x2, y2, znew.T, 10, lw=3, colors=["black"], levels=levels, linestyles="solid", linewidth=0.5)
    surf = ax2.plot_surface(x2, y2, znew.T, cmap=cmap, alpha = 0.99)
    fig.colorbar(surf, ax = ax2, shrink = 0.5, aspect = 10)

    return znew


def make_fit(make, x, y, M, p, X, Y):
    n = min([len(X), len(Y)])
    m = min([len(x), len(y)])
    A = np.array(list(zip(X, Y)))
    f = [make(A, i, p) for i in range(n)]
    B = np.array([[f[i](x[j], y[j]) for i in range(n)] for j in range(m)])
    N = optimize(B, M)
    # N = np.matmul(linalg.pinv(B), M)
    g = make(A, N, p)
    d = [g(x[i], y[i]) - M[i] for i in range(m)]
    print("d = ", d)

    return g


from scipy import optimize as opt


def optimize(B, M):
    M = np.matrix(M).T
    B = np.matrix(B)
    m = np.size(B, 0)
    n = np.size(B, 1)
    D = np.diag(np.ones(m))
    E = np.matrix(np.ones(m)).T
    Aeq = np.concatenate((B, -D, 0 * M), axis=1)
    beq = 0 * E
    A2 = np.concatenate((0 * B, D, -E), axis=1)
    A3 = np.concatenate((0 * B, -D, -E), axis=1)
    Aub = np.concatenate((A2, A3))
    bub = np.concatenate((M, -M))
    c = np.zeros(n + m + 1)
    c[-1] = 1
    print(Aub)
    print(bub)
    print(c)

    bounds = np.concatenate(
        (
            [(None, None) for x in range(n)],
            [(None, None) for x in range(m)],
            [(0, None)],
        )
    )
    res = opt.linprog(c, Aub, bub, Aeq, beq, bounds=bounds)
    if res.status != 0:
        print(res)
        raise (ArithmeticError("could not find optimal solution"))
    print("res.x = ", res.x)
    x = res.x[0:n]
    # x = np.matmul(linalg.pinv(B), M)
    return x


def example(make, x, y, M, p):
    xx = np.linspace(np.min(x) - 0.1, max(x) + 0.1, num=50)
    yy = np.linspace(np.min(y) - 0.1, max(y) + 0.1, num=50)
    A = np.array(list(zip(x, y)))
    pf = make(A, M, p)
    return plot(pf, x, y, xx, yy)


def example_fit(make, x, y, M, p, X, Y):
    pf = make_fit(make, x, y, M, p, X, Y)
    xx = np.linspace(min(x) - 0.1, max(x) + 0.1, num=50)
    yy = np.linspace(min(y) - 0.1, max(y) + 0.1, num=50)
    return plot(pf, x, y, xx, yy, X, Y)


def data():
    x = [1.0, 2, 3, 4, 4, 3, 2, 1, 1.5]
    y = [1.0, 1, 1, 1, 2, 2, 2.2, 2, 1.5]
    M = [0.0, 0, 0, 0, 1, 1, 1, 0, 0]

    X = [0.8, 2, 3, 4.2, 4.2, 3, 2, 0.8]
    Y = [0.8, 0.5, 0.5, 0.8, 2.2, 2.5, 2.3, 2.0]
    X = [0.9, 2, 3, 4.1, 4.1, 3, 2, 0.9, 0.8]
    Y = [1.0, 0.9, 0.9, 1, 2, 2.2, 2.3, 2, 1.5]
    return x, y, M, X, Y


if __name__ == "__main__":
    x, y, M, X, Y = data()

    # example(make_pkl, x, y, M, 2)
    # example(make_pmi, x, y, M, 3)
    example(make_pli, x, y, M, 0)
    # example(make_pba, x, y, M, 1)
    # example(make_pba, x, y, M, 1)
    # for i in range(len(x)):
    #    example(make_pba, x, y, i, 1)

    example_fit(make_pli, x, y, M, 1, X, Y)

    # fba(M, [1.9, 1.9], list(zip(x, y)), 0)

    plt.show()
