import numpy as np

class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.Z <= 0] = 0
        return dZ

class Softmax:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return self.A