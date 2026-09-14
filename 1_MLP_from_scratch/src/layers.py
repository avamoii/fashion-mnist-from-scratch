import numpy as np

class DenseLayer:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((n_out, 1))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        Z = np.dot(self.W, X) + self.b
        return Z

    def backward(self, dZ):
        m = self.X.shape[1]
        self.dW = (1.0 / m) * np.dot(dZ, self.X.T)
        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dX = np.dot(self.W.T, dZ)
        return dX