import numpy as np
import pygame
import random

class Neural_Net:
    # Structure: input (grid) -> 64 -> 64 -> 64 -> Q-Value
    episode = 5000
    batch = 32

    alpha = 0.01
    discount = 0.6
    default_move = 100
    
    exploration_decay = 0.05

    input_node = None
    hidden_node = 64
    output_node = 1

    key_W = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
    key_A = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
    key_S = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
    key_D = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
    keys = [key_W, key_A, key_S, key_D]

    def __init__(self, input_node):
        self.input_node = input_node
        self.exploration_rate = 0.95
        self.decision = 0
        self.record = []
        self.current_state = None
        self.move_left = self.default_move
        self.done = 0
        self.size = 1
        self.w1 = self.gen_weight(input_node, self.hidden_node)
        self.w2 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w3 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w4 = self.gen_weight(self.hidden_node, self.output_node)
        self.z1 = self.a1 = self.z2 = self.a2 = self.z3 = self.a3 = self.z4 = None

    def save_model(self):

        return
    
    def load_model(self, index):

        return
    
    def gen_weight(self, l1, l2):
        list = []
        for i in range(l1*l2):
            list.append(100*np.random.ranf()-50)
        return np.array(list).reshape(l1, l2)
    
    def reset(self):
        self.record.clear()
        self.done = False
        self.move_left = self.default_move
        return

    def setCurrentState(self, current_state):
        self.current_state = current_state
        return

    def getHidden(self):
        return (self.z1, self.a1, self.z2, self.a2, self.z3, self.a3, self.z4)
    
    def minus_move(self):
        if(self.move_left==0):
            self.done = True
        self.move_left-=1
        return
    
    def plus_move(self, size):
        if(size!=self.size):
            self.move_left+=self.default_move
            self.size = size
        return

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
    
    def relu(self, x):
        return x*(x>0)

    def relu_derivative(self, x):
        return x>0
    
    def calc_reward(self, dist):
        const = 0.00001
        return self.size
    
    def forward(self, input):
        self.z1 = np.dot(input, self.w1)
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3)
        self.a3 = self.relu(self.z3)
        self.z4 = np.dot(self.a3, self.w4)
        return self.z4
    
    def back_prop(self):
        if(len(self.record) == 0): return
        loss = self.loss_func(self.record)
        sample = random.sample(self.record, min(self.batch, len(self.record)))
        sample_len = float(len(sample))
        print("Loss: ", loss)
        dw1 = np.zeros_like(self.w1)
        dw2 = np.zeros_like(self.w2)
        dw3 = np.zeros_like(self.w3)
        dw4 = np.zeros_like(self.w4)
        # z1, a1, z2, a2, z3, a3, z4
        #  0   1   2   3   4   5   6
        for e in sample:
            y_true = e[1]+self.discount*self.forward(e[2])
            y_pred = self.forward(e[0])
            # Compute Gradient
            dz4 = y_pred-y_true
            dz3 = np.dot(dz4, self.w4.T)*self.relu_derivative(e[3][4])
            dz2 = np.dot(dz3, self.w3.T)*self.relu_derivative(e[3][2])
            dz1 = np.dot(dz2, self.w2.T)*self.relu_derivative(e[3][0])
            # Compute Weight
            dw4 += e[3][5].T.reshape(64, 1)*dz4
            dw3 += np.dot(e[3][3].T, dz3)
            dw2 += np.dot(e[3][1].T, dz2)
            dw1 += e[0].T.reshape(1200, 1)*dz1.reshape(1, 64)
        self.w1 -= dw1/sample_len*self.alpha
        self.w2 -= dw2/sample_len*self.alpha
        self.w3 -= dw3/sample_len*self.alpha
        self.w4 -= dw4/sample_len*self.alpha
        return
    
    def loss_func(self, sample): # MSE
        sum = 0.0
        cnt = 0.0
        for e in sample:
            y_true = e[1]+self.discount*self.forward(e[2])[0]
            y_pred = self.forward(e[0])[0]
            print(y_pred, y_true)
            sum+=(y_pred-y_true)*(y_pred-y_true)
            cnt+=1.0
        return float(sum/cnt)
     
    def pick_action(self, input, size, dist):
        mx = None
        for i in range(len(input)):
            q_value = self.forward(input[i])[0]
            if(mx == None or q_value>mx):
                mx = q_value
                self.decision = i
        if(np.random.normal(0, 1, 1)<=self.exploration_rate):
            self.decision = np.random.randint(4)
        self.minus_move()
        self.plus_move(size)
       # pygame.event.post(self.keys[self.decision])
        reward = self.calc_reward(dist)
        self.record_action(self.current_state, reward, input[self.decision], self.getHidden())
        self.setCurrentState(input[self.decision])
        return
    
    def record_action(self, max_q, reward, max_next_q, hidden):
        self.record.append((max_q, reward, max_next_q, hidden))
        return
