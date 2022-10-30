import numpy as np


class MSE():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def compute_loss(self):
        return np.mean(np.power(self.y_true-self.y_pred, 2))

    def derivative(self):
        return 2*(np.y_pred-np.y_true)/np.y_true.size
