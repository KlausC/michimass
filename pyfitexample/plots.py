#! /usr/bin/env python3

import warnings
import os

warnings.filterwarnings("ignore")
os.environ["XDG_SESSION_TYPE"] = "xcb"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # für 3d-plots

from scipy import linalg

from interpolation import *
from integration import *

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
    ax2.contour(
        x2,
        y2,
        znew.T,
        10,
        lw=3,
        colors=["black"],
        levels=levels,
        linestyles="solid",
        linewidth=0.5,
    )
    surf = ax2.plot_surface(x2, y2, znew.T, cmap=cmap, alpha=0.99)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

    return znew


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

    pa = (1.5, 1.6, -0.5)
    pb = (3.5, 1.9, -0.6)
    return x, y, M, X, Y, pa, pb


if __name__ == "__main__":
    x, y, M, X, Y = data()

    p = 1

    # make, p = make_pmi, 3
    make, p = make_pkl, 2
    # make = make_pli
    # make = make_pba

    example(make, x, y, M, p)
    example_fit(make, x, y, M, p, X, Y)

    # fba(M, [1.9, 1.9], list(zip(x, y)), 0)

    plt.show()