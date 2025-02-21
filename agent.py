import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from neural_network import FullyConnected
import random

class DoubleDQN:
    model_directory = "./model"

    episode = 1000000
    batch_size = 1

    learning_rate = 0.0005
    discount = 0.95
    default_move = 200
    
    epsilon_decay = 0.99999
    epsilon_min = 0.1
    
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.decision = 0
        self.experience = deque(maxlen=100000)
        self.current_state = None
        self.reward_list = []
        self.input_size = env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL
        self.output_size = len(env.keys)
        self.online_network = FullyConnected(self.input_size, self.output_size)
        self.target_network = FullyConnected(self.input_size, self.output_size)
        self.online_network.train()
        self.target_network.train()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
    
    def add_reward(self):
        self.reward_list.append(self.env.get_reward())
        return
    
    def save_reward(self):
        np.savetxt(f"{self.model_directory}/reward_hist.txt", np.array(self.reward_list), fmt="%s", delimiter=' ')
        return

    def get_model(self, model_name, train):
        self.online_network = torch.load(f"{self.model_directory}/{model_name}_o.pt", weights_only=False)
        self.target_network = torch.load(f"{self.model_directory}/{model_name}_t.pt", weights_only=False)
        if(train):
            self.online_network.train()
            self.target_network.train()
        else:
            self.online_network.eval()
            self.target_network.eval()
        return
    
    def save_model(self, model_name):
        torch.save(self.online_network, f"{self.model_directory}/{model_name}_o.pt")
        torch.save(self.online_network, f"{self.model_directory}/{model_name}_t.pt")
        print(f"Saved {model_name} successfully.")
        return

    def set_current_state(self, current_state):
        self.current_state = torch.tensor(current_state).reshape(self.input_size,)
        return
    
    def replay(self):
        if(len(self.experience)<self.batch_size): return
        sample = np.array(random.sample(self.experience, self.batch_size), dtype=object)
        with torch.no_grad():
            current_qvalue = self.online_network(torch.from_numpy(np.stack(sample[:,0])))
            next_qvalue = self.online_network(torch.from_numpy(np.stack(sample[:, 3])))
            target_qvalue = sample[:,2]+self.discount*np.max(self.target_network(torch.from_numpy(np.stack(sample[:,3]))).detach().numpy())*(1-sample[:,4])
        current_qvalue[np.arange(len(target_qvalue)), list(sample[:,1])] = target_qvalue
        self.online_network.fit(torch.from_numpy(np.stack(sample[:,0])), torch.from_numpy(current_qvalue), self.optimizer, self.loss_func)
        self.target_network.soft_update(self.online_network)
        return
    
    @torch.no_grad
    def pick_action(self, state):
        state = torch.tensor(state).reshape(self.input_size,)
        optimal_action = torch.argmax(self.online_network(state))
        print(self.online_network(state))
        random_action = np.random.randint(0, 4)
        action = np.random.choice([optimal_action, random_action], p=[1-self.epsilon, self.epsilon])
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        return action
    
    def record(self, action, next_state):
        next_state = torch.tensor(next_state).reshape(self.input_size,)
        self.experience.append((self.current_state, action, self.env.get_reward(), next_state, not self.env.plr.alive)) # s, a, r, s+1, d
        self.set_current_state(next_state)
        return

class DQN:
    model_directory = "./model"

    episode = 1000000
    batch_size = 1

    learning_rate = 0.0005
    discount = 0.95
    default_move = 200
    
    epsilon_decay = 0.999999
    epsilon_min = 0.1
    
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.decision = 0
        self.experience = deque(maxlen=100000)
        self.current_state = None
        self.reward_list = []
        self.input_size = env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL
        self.output_size = len(env.keys)
        self.online_network = FullyConnected(self.input_size, self.output_size)
        self.target_network = FullyConnected(self.input_size, self.output_size)
        self.online_network.train()
        self.target_network.train()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
    
    def add_reward(self):
        self.reward_list.append(self.env.get_reward())
        return
    
    def save_reward(self):
        np.savetxt(f"{self.model_directory}/reward_hist.txt", np.array(self.reward_list), fmt="%s", delimiter=' ')
        return

    def get_model(self, model_name, train):
        self.online_network = torch.load(f"{self.model_directory}/{model_name}_o.pt", weights_only=False)
        self.target_network = torch.load(f"{self.model_directory}/{model_name}_t.pt", weights_only=False)
        if(train):
            self.online_network.train()
            self.target_network.train()
        else:
            self.online_network.eval()
            self.target_network.eval()
        return
    
    def save_model(self, model_name):
        torch.save(self.online_network, f"{self.model_directory}/{model_name}_o.pt")
        torch.save(self.online_network, f"{self.model_directory}/{model_name}_t.pt")
        print(f"Saved {model_name} successfully.")
        return

    def set_current_state(self, current_state):
        self.current_state = torch.tensor(current_state).reshape(self.input_size,)
        return
    
    def replay(self):
        if(len(self.experience)<self.batch_size): return
        sample = np.array(random.sample(self.experience, self.batch_size), dtype=object)
        with torch.no_grad():
            current_qvalue = self.target_network(torch.from_numpy(np.stack(sample[:,0]))).detach().numpy()
            target_qvalue = sample[:,2]+self.discount*np.max(self.target_network(torch.from_numpy(np.stack(sample[:,3]))).detach().numpy())*(1-sample[:,4])
        current_qvalue[np.arange(len(target_qvalue)), list(sample[:,1])] = target_qvalue
        self.online_network.fit(torch.from_numpy(np.stack(sample[:,0])), torch.from_numpy(current_qvalue), self.optimizer, self.loss_func)
        self.target_network.soft_update(self.online_network)
        return
    
    @torch.no_grad
    def pick_action(self, state):
        state = torch.tensor(state).reshape(self.input_size,)
        optimal_action = torch.argmax(self.online_network(state))
        print(self.online_network(state))
        random_action = np.random.randint(0, 4)
        action = np.random.choice([optimal_action, random_action], p=[1-self.epsilon, self.epsilon])
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        return action
    
    def record(self, action, next_state):
        next_state = torch.tensor(next_state).reshape(self.input_size,)
        self.experience.append((self.current_state, action, self.env.get_reward(), next_state, not self.env.plr.alive)) # s, a, r, s+1, d
        self.set_current_state(next_state)
        return