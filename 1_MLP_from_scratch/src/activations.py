import numpy as np

class ReLU:
    def __init__(self):
        self.Z= None

    def forward(self,Z):
        self.Z = Z
        A=np.maximum(0,Z)
        return A

class Softmax:
    def __init__(self):
        self.A = None

    def forward(self,Z):
        exp_Z =np.exp(Z-np.max(Z,axis=0,keepdims=True))
        A=exp_Z/np.sum(exp_Z,axis=0,keepdims=True)
        self.A=A
        return A