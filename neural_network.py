import numpy as np
import utility as util
import os

class Neural_Network:
    model_directory = "./model"

    def __init__(self, structure, hidden_activation="relu", output_activation="relu", loss="mean_squared_error"):
        self.structure = structure
        self.layers = len(structure)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.build()
        pass

    def build(self):
        util.create_directory(self.model_directory)
        self.w = [self.gen_weight(self.structure[i-1], self.structure[i]) for i in range(1, self.layers, 1)]
        self.z = [None for i in range(self.layers+1)]
        self.b = [self.gen_bias(self.structure[i]) for i in range(self.layers)]
        self.a = [None for i in range(self.layers+1)]
        self.dw = [None for i in range(self.layers)]
        self.dz = [None for i in range(self.layers+1)]
        self.db = [None for i in range(self.layers+1)]
        self.setActivation(self.hidden_activation, True)
        self.setActivation(self.output_activation, False)
        self.setLoss(self.loss)
        return
    
    def setLoss(self, loss):
        if(loss == "mean_squared_error"):
            self.loss_func = self.mse_loss
            self.loss_dfunc = self.mse_derivative
        elif(loss == "categorical_crossentropy"):
            self.loss_func = self.cce_loss
            self.loss_dfunc = self.cce_derivative
        else:
            print("Invalid Loss Function.")
            exit(0)
        return
    
    def setActivation(self, activation, hidden):
        act_func = act_dfunc = None
        if(activation == "relu"):
            act_func = self.relu
            act_dfunc = self.relu_derivative
        elif(activation == "sigmoid"):
            act_func = self.sigmoid
            act_dfunc = self.sigmoid_derivative
        elif(activation == "softmax"):
            act_func = self.softmax
            act_dfunc = self.softmax_derivative
        elif(activation == "tanh"):
            act_func = self.tanh
            act_dfunc = self.tanh_derivative
        else:
            print("Invalid Activation Function.")
            exit(0)
        if(hidden == True):
            self.hidden_func = act_func
            self.hidden_dfunc = act_dfunc
        else: 
            self.output_func = act_func
            self.output_dfunc = act_dfunc
        return
    
    def load_model(self, folder_name, name):
        this_directory = f"./model/{folder_name}"
        if(not os.path.exists(this_directory)):
            print(f"{folder_name} does not exist.")
            return
        weight = [np.loadtxt(f"{this_directory}/{name}_{i+1}.txt", delimiter=" ", dtype=float) for i in range(self.layers-1)]
        self.copy_network(weight)
        return

    def save_model(self, folder_name, name):
        this_directory = f"./model/{folder_name}"
        util.create_directory(this_directory)
        np.set_printoptions(threshold=np.inf)
        for i in range(len(self.w)): np.savetxt(f"{this_directory}/{name}_{i+1}.txt", self.w[i], delimiter=" ", fmt="%s")
        print(f"Saved {folder_name} successfully.")
        return
    
    def copy_network(self, w):
        self.w = w.copy()
        return

    def update_network(self, weight, tau=0.001):
        for i in range(len(weight)): self.w[i] = tau*weight[i]+(1-tau)*self.w[i]
        return
    
    def gen_weight(self, l1, l2):
        return np.random.randn(l1, l2)*np.sqrt(1/l1)
    
    def gen_bias(self, size):
        return np.zeros(size)
    
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1-np.tanh(x)**2
    
    def relu(self, x):
        return x*(x>0).astype(float)

    def relu_derivative(self, x):
        return np.where(x>0, 1.0, 0.0)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig*(1-sig)
    
    def softmax(self, x):
        exp_x = np.exp(x-np.max(x))
        return exp_x/np.sum(exp_x)
    
    def softmax_derivative(self, x):
        if(self.loss == "categorical_crossentropy"): return 1
        softmax = self.softmax(x)
        return softmax*(1-softmax)
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2)
    
    def mse_derivative(self, y_pred, y_true):
        return 2*(y_pred-y_true)/(y_true.size)
    
    def cce_loss(self, y_pred, y_true):
        return -np.sum(y_pred*np.log(y_true+1e-9))
    
    def cce_derivative(self, y_pred, y_true):
        return y_pred-y_true
    
    def value_clipping(self, x, clip_value=1e5):
        return np.clip(x, -clip_value, clip_value)
    
    def back_prop(self, y_true, input, alpha=0.001):
        batch_size = y_true.shape[0]
        losses = self.loss_dfunc(self.a[self.layers-1], y_true)
        self.dz[self.layers-1] = losses*self.output_dfunc(self.z[self.layers-1])
        for i in range(self.layers-2, 0, -1): self.dz[i] = np.dot(self.dz[i+1], self.w[i].T)*self.hidden_dfunc(self.z[i])
        for i in range(1, self.layers): self.db[i] = np.sum(self.dz[i], axis=0)/batch_size
        self.dw[0] = np.dot(input.T, self.dz[1])/batch_size
        for i in range(1, self.layers-1): self.dw[i] = np.dot(self.a[i].T, self.dz[i+1])/batch_size
        for i in range(0, self.layers-1): self.w[i] -= alpha*self.dw[i]
        for i in range(1, self.layers): self.b[i] -= alpha*self.db[i]
        return
    
    def forward(self, input):
        self.z[1] = np.dot(input, self.w[0])+self.b[1]
        self.a[1] = self.hidden_func(self.z[1])
        for i in range(2, self.layers-1, 1):
            self.z[i] = np.dot(self.a[i-1], self.w[i-1])+self.b[i]
            self.a[i] = self.hidden_func(self.z[i])
        self.z[self.layers-1] = np.dot(self.a[self.layers-2], self.w[self.layers-2])+self.b[self.layers-1]
        self.a[self.layers-1] = self.output_func(self.z[self.layers-1])
        return self.a[self.layers-1]