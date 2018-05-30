import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import math

STEPS = 500
RES = 100

def f(x):
    return x*x*x - x

def fp(x):
    return 2*x*x - 1


def step(x, a):
    return x - a * f(x) / fp(x)


def go(x, a):
    for s in range(STEPS):
        x = step(x, a)

    return np.sign(x)

npts = RES
# xmin = math.sqrt(0.5)-.1; xmax = math.sqrt(0.5)+.1
xmin = -1.5; xmax = 1.5
ymin = 0.0; ymax = 1.5

xs = np.linspace(xmin, xmax, npts)
ys = np.linspace(ymin, ymax, npts)

tmp = np.zeros(npts, dtype=int)
zs = np.outer(tmp, tmp)  # A square array

for i in trange(npts):
    for j in range(npts):
        zs[i,j] = go(xs[i], ys[j])

fig = plt.gcf()
fig.set_size_inches(10,10)
plt.imshow(zs.T, origin='lower', interpolation='none')

plt.set_cmap('viridis')

plt.savefig('newton1d.png')
