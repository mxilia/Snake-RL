import numpy as np

class Neural_Network:

    def __init__(self, structure):
        self.structure = structure
        self.layers = len(structure)
        self.build()
        pass

    def build(self):
        self.w = [self.gen_weight(self.structure[i-1], self.structure[i]) for i in range(1, self.layers, 1)]
        self.z = [None for i in range(self.layers+1)]
        self.a = [None for i in range(self.layers+1)]
        self.dw = [None for i in range(self.layers)]
        self.dz = [None for i in range(self.layers+1)]
        self.da = [None for i in range(self.layers+1)]
        return
    
    def gen_weight(self, l1, l2):
        list = []
        for i in range(l1*l2):
            list.append(10*np.random.ranf()-5)
        return np.array(list).reshape(l1, l2)
    
    def relu(self, x):
        return x*(x>0)

    def relu_derivative(self, x):
        return x>0
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2)
    
    def mse_derivative(self, y_pred, y_true):
        return 2*np.mean(y_pred-y_true)
    
    def back_prop(self, sample, alpha=0.5):
        self.dz[4] = self.mse_derivative(self.z[self.layers-1], sample)
        return
    
    def forward(self, input):
        self.z[1] = np.dot(input, self.w[0])
        self.a[1] = self.relu(self.z[1])
        for i in range(2, self.layers, 1):
            self.z[i] = np.dot(self.a[i-1], self.w[i-1])
            self.a[i] = self.relu(self.z[i])
        return self.z[self.layers-1]
    