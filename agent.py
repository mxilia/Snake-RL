import numpy as np
import pygame

class Neural_Net:
    # Structure: input (grid) -> 64 -> 64 -> 64 -> Q-Value
    episode = 50

    alpha = 0.05
    exploration_rate = 0.95
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
        self.decision = 0
        self.record = []
        self.current_state = None
        self.current_reward = 0
        self.remaining_move = 100
        self.w1 = self.gen_weight(input_node, self.hidden_node)
        self.w2 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w3 = self.gen_weight(self.hidden_node, self.hidden_node)
        self.w4 = self.gen_weight(self.hidden_node, self.output_node)

    def setCurrentState(self, current_state):
        self.current_state = current_state
        return

    def reset(self):
        self.record.clear()
        return

    def gen_weight(self, l1, l2):
        list = []
        for i in range(l1*l2):
            list.append(np.random.ranf())
        return np.array(list).reshape(l1, l2)

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
    
    def relu(self, x):
        return x*(x>0)
    
    def forward(self, input):
        z1 = np.dot(input, self.w1)
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.w2)
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.w3)
        a3 = self.relu(z3)
        z4 = np.dot(a3, self.w4)
        return z4
    
    def back_prop(self):
        
        return
    
    def loss_func(self):
        
        return
    
    def pick_action(self, input):
        if(np.random.normal(0, 1, 1)<=self.exploration_rate or True):
            mx = None
            for i in range(len(input)):
                q_value = self.forward(input[i])[0]
                if(mx == None or q_value>mx):
                    mx = q_value
                    self.decision = i
        else:
            self.decision = np.random.randint(4)
        pygame.event.post(self.keys[self.decision])
        self.record_action(self.current_state, self.decision, self.current_reward, input[self.decision])
        self.setCurrentState(input[self.decision])
        return
    
    def record_action(self, current_state, action, reward, next_state):
        self.record.append((current_state, action, reward, next_state))
        return

    
