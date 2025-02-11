import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from neural_network import Neural_Network
import random

class DQN:
    # Structure: input (grid) -> 64 -> 64 -> 64 -> 4 Q-Value for wasd
    episode = 10000
    batch_size = 32

    alpha = 0.01
    discount = 0.99
    default_move = 100
    
    epsilon_decay = 0.99
    epsilon_min = 0.1

    hidden_node = 64
    output_node = 4

    reward_list = []

    def __init__(self, input_node):
        self.input_node = input_node
        self.epsilon = 1.0
        self.decision = 0
        self.experience = deque(maxlen=100000)
        self.current_state = None
        self.reward = 0
        self.prev_size = 1
        self.done = False
        self.online_network = Neural_Network((input_node, self.hidden_node, self.hidden_node, self.hidden_node, self.output_node))
        self.target_network = Neural_Network((input_node, self.hidden_node, self.hidden_node, self.hidden_node, self.output_node))
    
    def reset(self):
        self.done = False
        self.reward_list.append(self.reward)
        self.reward = 0
        self.prev_size = 1
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        return

    def setCurrentState(self, current_state):
        self.current_state = current_state
        return
    
    def replay(self):
        if(len(self.experience)<self.batch_size): return
        sample = np.array(random.sample(self.experience, self.batch_size), dtype=object)
        current_qvalue = np.array(self.online_network.forward(np.stack(sample[:,0])))
        target_qvalue = sample[:,2]+self.discount*np.max(self.target_network.forward(np.stack(sample[:,3])))*(1-sample[:,4])
        current_qvalue[np.arange(len(target_qvalue)),list(sample[:,1])] = target_qvalue
        self.online_network.back_prop(current_qvalue, np.stack(sample[:,0]), self.batch_size, alpha=self.alpha)
        self.target_network.update_network(self.online_network.w)
        return
    
    def update_reward(self, size, status):
        self.done = status
        self.reward-=0.1
        if(size>self.prev_size):
            self.prev_size = size
            self.reward=max(0, self.reward+100)
        if(self.done):
            self.reward-=self.input_node-2*size
        return
     
    def pick_action(self, state):
        optimal_action = np.argmax(self.online_network.forward(np.array([state]))[0])
        random_action = np.random.randint(0, 4)
        action = np.random.choice([optimal_action, random_action], p=[1-self.epsilon, self.epsilon])
        return action
    
    def record(self, action, next_state):
        self.experience.append((self.current_state, action, self.reward, next_state, self.done)) # s, a, r, s+1, d
        self.setCurrentState(next_state)
        return
    
    def result(self):
        plt.figure(figsize=(6,4), dpi=100)
        plt.plot(self.reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
        return
