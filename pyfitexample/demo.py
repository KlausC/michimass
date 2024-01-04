#! /usr/bin/env python3

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import interp2d, RectBivariateSpline

from scipy.interpolate import bisplrep, bisplev

from numpy.testing import assert_allclose

x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 7.51, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2 + 2.*yy**2)
f = interp2d(x, y, z, kind='cubic')

def plot(f, xnew, ynew):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    znew = f(xnew, ynew)

    ax1.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')

    im = ax2.imshow(znew)
    plt.colorbar(im, ax=ax2)

    plt.show()
    return znew

xnew = np.arange(-5.01, 5.01, 1e-2)
ynew = np.arange(-5.01, 7.51, 1e-2)
# znew_i = plot(f, xnew, ynew)

r = RectBivariateSpline(x, y, z.T)

rt = lambda xnew, ynew: r(xnew, ynew).T
znew_r = plot(rt, xnew, ynew)

# TestSmoothBivariateSpline::test_integral
from scipy.interpolate import SmoothBivariateSpline, LinearNDInterpolator

x = np.array([1,1,1,2,2,2,4,4,4])
y = np.array([1,2,3,1,2,3,1,2,3])
z = np.array([0,7,8,3,4,7,1,3,4])

# Now, use the linear interpolation over Qhull-based triangulation of data:

xy = np.c_[x, y]   # or just list(zip(x, y))
lut2 = LinearNDInterpolator(xy, z)

X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)

# The result is easy to understand and interpret:

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(X, Y, lut2(X, Y))
ax.scatter(x, y, z,  'o', color='k', s=48)

plt.show()

# Note that bisplrep does something different! It may place spline knots outside of the data.

# For illustration, consider the same data from the previous example:

tck = bisplrep(x, y, z, kx=1, ky=1, s=0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xx = np.linspace(min(x), max(x))
yy = np.linspace(min(y), max(y))
X, Y = np.meshgrid(xx, yy)
Z = bisplev(xx, yy, tck)
Z = Z.reshape(*X.shape).T

ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
ax.scatter(x, y, z,  'o', color='k', s=48)

plt.show()

#assert_allclose(znew_r, atol=1e-14)

