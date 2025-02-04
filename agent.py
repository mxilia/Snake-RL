import numpy as np
import pygame
import random

class Neural_Net:
    # Structure: input (grid) -> 64 -> 64 -> 64 -> Q-Value
    episode = 500
    batch = 32

    alpha = 0.05
    discount = 0.90
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

    def gen_weight(self, l1, l2):
        list = []
        for i in range(l1*l2):
            list.append(np.random.ranf())
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

    def diff_relu(self, x):
        return x>0
    
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
        sample = random.sample(self.record, min(self.batch, len(self.record)))
        loss = self.loss_func(sample)
        for e in sample:
            hidden = e[3]
            return
        return
    
    def loss_func(self, sample): # MSE
        sum = 0.0
        for e in sample:
            y = e[1]+self.discount*self.forward(e[2])[0]
            q_value = self.forward(e[0])[0]
            sum+=(y-q_value)*(y-q_value)
        return sum/float(len(sample))
     
    def pick_action(self, input, size):
        if(np.random.normal(0, 1, 1)<=self.exploration_rate):
            mx = None
            for i in range(len(input)):
                q_value = self.forward(input[i])[0] 
                if(mx == None or q_value>mx):
                    mx = q_value
                    self.decision = i
        else:
            self.decision = np.random.randint(4)
        self.minus_move()
        self.plus_move(size)
        pygame.event.post(self.keys[self.decision])
        reward = self.size*(float(self.move_left)/float(self.default_move/2))
        self.record_action(self.current_state, reward, input[self.decision], self.getHidden())
        self.setCurrentState(input[self.decision])
        return
    
    def record_action(self, current_state, reward, next_state, hidden):
        self.record.append((current_state, reward, next_state, hidden))
        return

    
