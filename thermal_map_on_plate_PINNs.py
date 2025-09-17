"""Thermal Map on Plate"""

import deepxde as dde
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Geometry

geom = dde.geometry.Rectangle([0 ,0], [3, 3])

from re import X
# define PDE steady state heat eq

def pde(x, u):
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)

# heat sourcce at center
    x0 = 1.5
    y0 = 1.5
    sigma = 0.25
    A = 20

# x is a tensor
    dx2 = (x[:, 0:1] - x0) ** 2
    dy2 = (x[:, 1:2] - y0) ** 2

    Q = A * tf.exp(-(dx2 + dy2) / (2 * sigma**2))

    return u_xx + u_yy + Q

# def BC

def boundary(x, on_boundary):
    return on_boundary

bc = dde.DirichletBC(geom, lambda x: 0, boundary)

# Data & Net

data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain = 3000,
    num_boundary = 200,
    num_test = 1000,
)

net = dde.maps.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

# Train the model

model.compile("adam", lr=0.001)
LossHistory, train_state = model.train(epochs=10000)

#plotting

nx = ny = 100
x = np.linspace(0, 3, nx)
y = np.linspace(0, 3, ny)
X, Y = np.meshgrid(x, y)
XY = np.vstack((X.flatten(), Y.flatten())).T
u = model.predict(XY)
U = u.reshape((nx, ny))

plt.figure(figsize=(6,5))
plt.contourf(X, Y, U, 20)
plt.colorbar(label = "Temperature")
plt.title("Thermal Map on Plate/Center Hotspot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
