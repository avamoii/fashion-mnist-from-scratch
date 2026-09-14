import numpy as np

class DenseLayer:
    def __init__(self, n_in,n_out):
        self.w =np.random.randn(n_out, n_in)*np.sqrt(2.0/n_in)
        self.b =np.zeros((n_out,1))

        self.X =None
        self.dw=None
        self.db =None
    def forward(self,X):
        self.X=X
        Z=np.dot(self.W,X)+self.b
        return Z