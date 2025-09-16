# -*- coding: utf-8 -*-
"""Thermal Model Exract.ipynb"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

L = 0.07
H = 0.002

k_fr4 = 0.3
k_cu = 400.0

cu_x0, cu_x1 = 0.02, 0.03
cu_y0, cu_y1 = 0.02, 0.03

Q_cu = 5e7

geom = dde.geometry.Rectangle([0, 0], [L, H])

def kappa(X):
  xcoord = X[:, 0:1]
  ycoord = X[:, 1:2]

  # boolean mask on copper
  inside_X = tf.math.logical_and(tf.math.logical_and(xcoord >= cu_x0, xcoord <= cu_x1), tf.math.logical_and(ycoord >= cu_y0, ycoord <= cu_y1))
  inside_Y = tf.math.logical_and(tf.math.logical_and(ycoord >= cu_y0, ycoord <= cu_y1), tf.math.logical_and(xcoord >= cu_x0, xcoord <= cu_x1))
  inside_cu = tf.math.logical_or(inside_X, inside_Y)

  k = tf.where(inside_cu, k_cu, k_fr4)
  return k

def pde(X, U):
  U_X = dde.grad.jacobian(U, X, i=0, j=0)
  U_Y = dde.grad.jacobian(U, X, i=0, j=1)

  k = kappa(X)
  k_U_X = dde.grad.jacobian(k * U, X, i=0, j=0)
  k_U_Y = dde.grad.jacobian(k * U, X, i=0, j=1)

  div_X = k_U_X
  div_Y = k_U_Y

  xcoord = X[:, 0:1]
  ycoord = X[:, 1:2]

  in_cu = tf.math.logical_and(tf.math.logical_and(xcoord >= cu_x0, xcoord <= cu_x1), tf.math.logical_and(ycoord >= cu_y0, ycoord <= cu_y1))

  Q = tf.where(in_cu, Q_cu, 0)

  return div_X + div_Y + Q

bc = dde.icbc.DirichletBC(geom, lambda X: 0, lambda _, on_boundary: on_boundary)

data = dde.data.PDE(geom, pde, bc, 15000, 500)

net = dde.maps.FNN([2] + [100] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=10000)

model.compile("L-BFGS")
model.train()

nx, ny = 200, 60
xs = np.linspace(0, L, nx)
ys = np.linspace(0, cu_y1, ny) # Adjusted the upper limit of ys to include the copper region
X, Y = np.meshgrid(xs, ys)
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
U = model.predict(X_star)
U = U.reshape(ny, nx)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, U, cmap="jet", shading='gouraud')
plt.colorbar()
plt.title("Predicted Solution")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# coordinates from X_star to create the mask
mask = np.logical_and(np.logical_and(X_star[:, 0] >= cu_x0, X_star[:, 0] <= cu_x1), np.logical_and(X_star[:, 1] >= cu_y0, X_star[:, 1] <= cu_y1))

T_cu_avg = U.ravel()[mask].mean() # flattened mask
T_amb = 0
P_total = Q_cu * ((cu_x1 - cu_x0) * (cu_y1 - cu_y0))
R_th = (T_cu_avg - T_amb) / P_total
print("Approx. R_th: ", R_th)
