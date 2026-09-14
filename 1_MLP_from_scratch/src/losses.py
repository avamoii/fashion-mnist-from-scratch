import numpy as np

class CrossEntropyLoss:
    def forward(self, Y_hat, Y):
        m = Y.shape[1]
        Y_hat_clipped = np.clip(Y_hat, 1e-15, 1 - 1e-15)
        loss = -(1.0 / m) * np.sum(Y * np.log(Y_hat_clipped))
        return loss

    def backward(self, Y_hat, Y):
  
        return Y_hat - Y