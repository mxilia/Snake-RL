import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from neural_network import Neural_Network
import utility as util
import random

class DQN:
    # Structure: input (grid) -> 64 -> 64 -> 64 -> 4 Q-Value for wasd
    episode = 1000
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
        self.default_move = 150
        self.time = 0
        self.remaining_move = self.default_move
        self.done = False
        self.online_network = Neural_Network((input_node, self.hidden_node, self.hidden_node, self.hidden_node, self.output_node))
        self.target_network = Neural_Network((input_node, self.hidden_node, self.hidden_node, self.hidden_node, self.output_node))
    
    def reset(self):
        self.done = False
        self.remaining_move = self.default_move
        self.reward_list.append(self.reward)
        if(self.reward<0 and self.epsilon<0.5): self.epsilon = 0.5
        self.reward = 0
        self.prev_size = 1
        return

    def get_model(self):
        return
        epoch = int(open("./model/epoch.txt", "r").read())
        online_directory = f"./model/epoch_{epoch}/online.txt"
        target_directory = f"./model/epoch_{epoch}/target.txt"
        epsilon_directory = f"./model/epoch_{epoch}/epsilon.txt"
        #with open(online_directory, )

        with open(epsilon_directory, "r") as file:
            file.write(self.epsilon)
        return
    
    def save_model(self):
        epoch = int(open("./model/epoch.txt", "r").read())+1
        online_directory = f"./model/epoch_{epoch}/online.txt"
        target_directory = f"./model/epoch_{epoch}/target.txt"
        epsilon_directory = f"./model/epoch_{epoch}/epsilon.txt"
        util.file_create(online_directory)
        util.file_create(target_directory)
        util.file_create(epsilon_directory)
        online_weight = np.array(self.online_network.getWeight(), dtype=object)
        target_weight = np.array(self.target_network.getWeight(), dtype=object)
        np.set_printoptions(threshold=np.inf)
        np.savetxt(online_directory, online_weight, fmt="%s")
        np.savetxt(target_directory, target_weight, fmt="%s")
        with open(epsilon_directory, "w") as file:
            file.write(str(self.epsilon))
        if(input() == "n"): return
        file = open("./model/epoch.txt", "w")
        file.write(str(epoch))
        print(f"Saved epoch_{epoch} successfully.")
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
        self.remaining_move-=1
        self.reward+=1
        if(size>self.prev_size):
            self.prev_size = size
            self.reward=max(0, self.reward+100)
            self.remaining_move+=self.default_move
        if(self.remaining_move==0):
            self.done = True
        if(self.done):
            self.reward-=self.input_node-2*size+self.time
        return
     
    def pick_action(self, state):
        optimal_action = np.argmax(self.online_network.forward(np.array([state]))[0])
        random_action = np.random.randint(0, 4)
        action = np.random.choice([optimal_action, random_action], p=[1-self.epsilon, self.epsilon])
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
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
