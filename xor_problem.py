import numpy as np

from network import Network
from layers import Dense, Sigmoid, Tanh
from losses import MSE

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Dense(2, 3))
net.add(Sigmoid())
net.add(Dense(3, 1))
net.add(Sigmoid())

# train
net.complie_loss(MSE())
net.fit(x_train, y_train, epochs=1000, learning_rate=1)

# test
out = net.predict(x_train)
print(out)