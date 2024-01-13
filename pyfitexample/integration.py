from interpolation import make_pli as make

import numpy as np

# given n coordinates x an y, and m * n niveaus z[m, n] at all coordinate points.
# given two endpoints of a line segment L: (xa, ya, za) and (xb, yb, zb)

# use `make` to define a smooth functions f(m, x, y) -> z to assign a niveau
# for to every m to each coordinate point (x, y) along L.

# Plot the niveau lines along L.
# Approximate the area between consecutive niveau lines (layers).

# parameter for make function
p = 0


def make_niveau(nr, X, Y, Z):
    Z = np.array(Z)
    n = len(X)
    m = np.size(Z, 1)
    if n != len(Y) or n != np.size(Z, 0):
        raise np.exceptions.AxisError("number of points not consistent")
    if not (0 <= nr < m):
        raise ValueError("nr must be in 0...m")
    for i in range(n):
        for j in range(m - 1):
            if Z[i][j] < Z[i][j + 1]:
                raise ValueError("all niveaus must be sorted")

    M = [Z[i][nr] for i in range(n)]
    A = np.array(list(zip(X, Y)))
    return make(A, M, p)


def make_niveau_section(nr, X, Y, Z, pa, pb):
    f = make_niveau(nr, X, Y, Z)
    xa, ya, za = pa
    xb, yb, zb = pb
    xc = xb - xa
    yc = xb - xa
    return lambda t: f(xa + t * xc, ya + t * yc)


def make_niveau_over(nr, X, Y, Z, pa, pb):
    f = make_niveau(nr, X, Y, Z)
    xa, ya, za = pa
    xb, yb, zb = pb
    xc = xb - xa
    yc = xb - xa
    zc = zb - za
    return lambda t: max(f(xa + t * xc, ya + t * yc), za + t * zc)


def integrate_sum(nr, X, Y, Z, pa, pb):
    f = make_niveau_over(nr, X, Y, Z, pa, pb)
    n = 1000
    return sum(f(t) for t in np.linspace(0.5 / n, 1.0, num=n))


def integrate_level(nr, X, Y, Z, pa, pb):
    s = integrate_sum(nr, X, Y, Z, pa, pb) - integrate_sum(nr, X, Y, Z, pa, pb)
    return s
