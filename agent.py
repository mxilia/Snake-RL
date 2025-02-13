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

    alpha = 0.005
    discount = 0.99
    default_move = 200
    
    epsilon_decay = 0.99
    epsilon_min = 0.1

    hidden_node = 64
    output_node = 4
    
    def __init__(self, input_node, env):
        self.input_node = input_node
        self.env = env
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
        self.reward_list = []
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
        epoch = int(open("./model/epoch.txt", "r").read())
        if(epoch == 0): return
        epoch_directory = f"./model/epoch_{epoch}"
        stats_file = f"./model/epoch_{epoch}/stats.txt"
        layers = None
        with open(stats_file, "r") as file:
            self.epsilon = float(file.readline())
            layers = int(file.readline())
        online_weight = [np.loadtxt(epoch_directory+f"/online_{i+1}.txt", delimiter=" ", dtype=float) for i in range(layers)]
        target_weight = [np.loadtxt(epoch_directory+f"/target_{i+1}.txt", delimiter=" ", dtype=float) for i in range(layers)]
        self.online_network.copy_network(online_weight)
        self.online_network.copy_network(target_weight)
        return
    
    def save_model(self):
        if(len(self.reward_list)<self.episode):
            print("Not enough episodes.")
            return
        epoch = int(open("./model/epoch.txt", "r").read())+1
        epoch_directory = f"./model/epoch_{epoch}"
        stats_file = f"./model/epoch_{epoch}/stats.txt"
        reward_file = f"./model/epoch_{epoch}/reward_list.txt"
        util.create_directory(epoch_directory)
        online_weight = self.online_network.getWeight()
        target_weight = self.target_network.getWeight()
        reward_list = np.array(self.reward_list).reshape(len(self.reward_list))
        np.set_printoptions(threshold=np.inf)
        for i in range(len(online_weight)): np.savetxt(epoch_directory+f"/online_{i+1}.txt", online_weight[i], delimiter=" ", fmt="%s")
        for i in range(len(target_weight)): np.savetxt(epoch_directory+f"/target_{i+1}.txt", target_weight[i], delimiter=" ", fmt="%s")
        with open(stats_file, "w") as file: file.write(str(self.epsilon)+"\n"+str(len(online_weight)))
        np.savetxt(reward_file, reward_list, delimiter=" ", fmt="%s")
        file = open("./model/epoch.txt", "w")
        file.write(str(epoch))
        print(f"Saved epoch_{epoch} successfully.")
        return

    def setCurrentState(self, current_state):
        self.current_state = current_state
        return
    
    def replay(self):
        self.reward = self.env.getReward()
        if(len(self.experience)<self.batch_size): return
        sample = np.array(random.sample(self.experience, self.batch_size), dtype=object)
        current_qvalue = np.array(self.online_network.forward(np.stack(sample[:,0])))
        target_qvalue = sample[:,2]+self.discount*np.max(self.target_network.forward(np.stack(sample[:,3])))*(1-sample[:,4])
        current_qvalue[np.arange(len(target_qvalue)),list(sample[:,1])] = target_qvalue
        self.online_network.back_prop(current_qvalue, np.stack(sample[:,0]), self.batch_size, alpha=self.alpha)
        self.target_network.update_network(self.online_network.w)
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
