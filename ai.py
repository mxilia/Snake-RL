import numpy as np

class Neural_Net:
    # Structure: input (grid) -> 16 -> 16 -> 16 -> output (wasd)
    alpha = 0.05
    exploration_rate = 0.95
    exploration_decay = 0.05

    input_node = 0
    hidden_node = 16
    output_node = 4
    move = 20
    
    decision = " "
    action = []

    def __init__(self, input_node):
        self.input_node = input_node
        self.w1 = self.gen_weight(input_node, self.hidden_node)
        self.w2 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w3 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w4 = self.gen_weight(self.hidden_node, self.output_node)

    def reset(self):
        self.action.clear()
        return

    def gen_weight(self, l1, l2):
        list = []
        for i in range(l1*l2):
            list.append(np.random.randn())
        return np.array(list).reshape(l1, l2)

    def sigmoid(self, x):
        return (1/(1 + np.exp(-x)))
    
    def forward(self, input):
        z1 = np.dot(input, self.w1)
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.w2)
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.w3)
        a3 = self.sigmoid(z3)
        z4 = np.dot(a3, self.w4)
        a4 = self.sigmoid(z4)
        return a4
    
    def back_prop(self):

        return
    
    def loss_func():

        return

    def predict(self, input, sol):
        policy = self.forward(input)
        self.action.append((policy, sol))
        mx = -1.0
        for i in range(self.output_node):
            if(policy[0][i]>mx):
                mx = policy[0][i]
                self.decision = i
        return

    
