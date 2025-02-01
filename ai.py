import numpy as np
import pygame

class Neural_Net:
    # Structure: input (grid) -> 16 -> 16 -> 16 -> output (wasd)
    episode = 50

    alpha = 0.05
    exploration_rate = 0.95
    exploration_decay = 0.05

    input_node = 0
    hidden_node = 16
    output_node = 4
    
    decision = 0
    record = []

    key_W = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
    key_A = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
    key_S = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
    key_D = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
    keys = [key_W, key_A, key_S, key_D]

    def __init__(self, input_node):
        self.input_node = input_node
        self.w1 = self.gen_weight(input_node, self.hidden_node)
        self.w2 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w3 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w4 = self.gen_weight(self.hidden_node, self.output_node)

    def load_version(index):
        return
    
    def save_version():
        return
    def reset(self):
        self.record.clear()
        return

    def gen_weight(self, l1, l2):
        list = []
        for i in range(l1*l2):
            list.append(np.random.randn())
        return np.array(list).reshape(l1, l2)

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
    
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
    
    def loss_func(self):

        return
    
    def pick_action(self, input):
        policy = self.forward(input)
        mx = -1.0
        for i in range(self.output_node):
            if(policy[0][i]>mx):
                mx = policy[0][i]
                self.decision = i
        pygame.event.post(self.keys[self.decision])
        return
    
    def record_action(self, current_state, next_state):
        self.record.append(current_state, next_state)
        return

    
