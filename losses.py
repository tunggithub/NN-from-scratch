import numpy as np


class MSE():
    def compute_loss(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def derivative(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size
