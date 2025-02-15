import numpy as np

class Neural_Network:

    def __init__(self, structure, activation="relu"):
        self.structure = structure
        self.layers = len(structure)
        self.build(activation)
        pass

    def build(self, activation):
        self.w = [self.gen_weight(self.structure[i-1], self.structure[i]) for i in range(1, self.layers, 1)]
        self.z = [None for i in range(self.layers+1)]
        self.a = [None for i in range(self.layers+1)]
        self.dw = [None for i in range(self.layers)]
        self.dz = [None for i in range(self.layers+1)]
        if(activation == "relu"):
            self.act_func = self.relu
            self.deriv_act_func = self.relu_derivative
        elif(activation == "sigmoid"):
            self.act_func = self.sigmoid
            self.deriv_act_func = self.sigmoid_derivative
        return
    
    def getWeight(self):
        return self.w
    
    def copy_network(self, w):
        self.w = w.copy()
        return

    def update_network(self, weight, tau=0.001):
        for i in range(len(weight)): self.w[i] = tau*weight[i]+(1-tau)*self.w[i]
        return
    
    def gen_weight(self, l1, l2):
        return np.random.normal(0.5, 2.5, size=(l1, l2))
    
    def relu(self, x):
        return x*(x>0).astype(float)

    def relu_derivative(self, x):
        return np.where(x>0, 1.0, 0.0)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2)
    
    def mse_derivative(self, y_pred, y_true):
        return 2*(y_pred-y_true)/(y_true.size)
    
    def value_clipping(self, x, clip_value=1e7):
        return np.clip(x, -clip_value, clip_value)
    
    def back_prop(self, sample, input, batch_size, alpha=0.5):
        self.dz[self.layers-1] = self.value_clipping(self.mse_derivative(self.z[self.layers-1], sample))
        for i in range(self.layers-2, 0, -1): self.dz[i] = np.dot(self.dz[i+1], self.w[i].T)*self.deriv_act_func(self.z[i])
        self.dw[0] = np.dot(input.T, self.dz[1])/batch_size
        for i in range(1, self.layers-1): self.dw[i] = np.dot(self.a[i].T, self.dz[i+1])/batch_size
        for i in range(0, self.layers-1): self.w[i] -= alpha*self.dw[i]
        return
    
    def forward(self, input):
        self.z[1] = np.dot(input, self.w[0])
        self.a[1] = self.act_func(self.z[1])
        for i in range(2, self.layers, 1):
            self.z[i] = np.dot(self.a[i-1], self.w[i-1])
            self.a[i] = self.act_func(self.z[i])
        return self.z[self.layers-1]
    