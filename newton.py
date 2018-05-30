import torch

from torch.autograd import Variable

import torch.optim as optim
import torch.nn as nn

import cmath as cm
import numpy as np
import matplotlib.pyplot as plt

import random, math
from tqdm import trange

STEPS = 100
RES = 100


class Net(nn.Module):

    def __init__(self, r0, c0):
        super(Net, self).__init__()
        self.r = nn.Parameter(torch.FloatTensor([r0]))
        self.c = nn.Parameter(torch.FloatTensor([c0]))

    def forward(self):

        cube_r = -3 * self.c * self.c * self.r + self.r * self.r * self.r
        cube_c = 3 * self.c * self.r * self.r - self.c * self.c * self.c

        fin_r = cube_r - 1
        fin_c = cube_c

        return (fin_r, fin_c)
def loss_func(o):
    return o[0] * o[0] + o[1] * o[1]

# Difficult bits are thanks to http://sam-dolan.staff.shef.ac.uk/mas212/notebooks/Newton_Fractal.html

roots = [cm.exp(2.0*k*cm.pi*1j / 3.0) for k in range(3)]  # Known roots

def get_value(x, y):
    eps = 0.1

    # not quite the same as newton's method, but close
    net = Net(x, y)
    optimizer = optim.SGD(net.parameters(), lr=0.5)

    for k in range(STEPS):
        optimizer.zero_grad()  # zero the gradient buffers
        output = net()
        loss = loss_func(output)
        loss.backward()

        optimizer.step()

    p = list(net.parameters())
    z = complex(net.r.data[0], net.c.data[0])

    val  = 4
    for i in range(3):
        if abs(z - roots[i]) < eps:
            val = i+1

    return val

def get_value_manual(x, y):
    eps = 0.5

    net = Net(x, y)

    loss = loss_func(net())
    loss.backward()

    lr = 0.01

    for k in range(STEPS):
        loss = loss_func(net())

        # net.r.grad.data.zero_()
        # net.c.grad.data.zero_()

        g = torch.autograd.grad(loss, [net.r, net.c], create_graph=True)

        h = np.zeros((2,2))

        h[0, 0] = torch.autograd.grad(g[0], [net.r], create_graph=True)[0].data[0]
        h[0, 1] = torch.autograd.grad(g[1], [net.r], create_graph=True)[0].data[0]
        h[1, 0] = torch.autograd.grad(g[0], [net.c], create_graph=True)[0].data[0]
        h[1, 1] = torch.autograd.grad(g[1], [net.c], create_graph=True)[0].data[0]

        gnp = np.asarray([[g[0].data[0]],[g[1].data[0]]])
        hinv = np.linalg.pinv(h)

        # print(gnp)
        # print(h)
        # print(hinv)
        # print()

        gnp = hinv.dot(gnp)

        net.r.data -= lr * gnp[0,0].data[0]
        net.c.data -= lr * gnp[1,0].data[0]

    p = list(net.parameters())
    z = complex(net.r.data[0], net.c.data[0])

    val  = 4
    for i in range(3):
        if abs(z - roots[i]) < eps:
            val = i+1

    return val

def hessian_norm(x, y):
    eps = 0.5

    net = Net(x, y)

    loss = loss_func(net())
    loss.backward()

    lr = 0.01

    loss = loss_func(net())

    # net.r.grad.data.zero_()
    # net.c.grad.data.zero_()

    g = torch.autograd.grad(loss, [net.r, net.c], create_graph=True)

    h = np.zeros((2, 2))

    a = torch.autograd.grad(g[0], [net.r], create_graph=True)[0].data[0]
    b = torch.autograd.grad(g[1], [net.r], create_graph=True)[0].data[0]
    c = torch.autograd.grad(g[0], [net.c], create_graph=True)[0].data[0]
    d = torch.autograd.grad(g[1], [net.c], create_graph=True)[0].data[0]

    return a*a+b*b+c*c+d*d


def grad_norm(x, y):
    net = Net(x, y)

    net = Net(x, y)
    optimizer = optim.SGD(net.parameters(), lr=0.5)

    optimizer.zero_grad()  # zero the gradient buffers
    output = net()
    loss = loss_func(output)
    loss.backward()

    a = net.r.grad.data[0]
    b = net.c.grad.data[0]

    return a*a + b*b

def f(x, y):
    net = Net(x, y)

    return loss_func(net()).data[0]

npts = RES; xmin = -1.5; xmax = 1.5
xs = np.linspace(xmin, xmax, npts)
tmp = np.zeros(npts, dtype=float)
zs = np.outer(tmp, tmp)  # A square array
for i in trange(npts):
    for j in range(npts):
        zs[i,j] = math.log(hessian_norm(xs[i], xs[j]))

fig = plt.gcf()
fig.set_size_inches(10,10)
plt.imshow(zs.T, origin='lower', interpolation='none')

plt.set_cmap('viridis')
plt.colorbar()

plt.savefig('newton.png')



