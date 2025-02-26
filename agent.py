import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random

import utility as util 
from neural_network import ConvoNN
from neural_network import DuelingNetWork

class DQN:
    
    def __init__(self, input_dim, output_dim, model_name="normal_dqn"):
        self.model_directory = f"./{model_name}"
        util.create_directory(self.model_directory)

        self.num_episode = 100000

        self.epsilon = 1.0
        self.epsilon_decay = 0.999999
        self.epsilon_min = 0.02

        self.discount = 0.99
        self.learning_rate = 0.0002

        self.batch_size = 16
        self.memory_size = 100000
        self.target_net_update_int = 500
        self.time = 0

        self.memory = deque(maxlen=self.memory_size)
        self.reward_hist = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.online_network = ConvoNN(input_dim, output_dim)
        self.target_network = ConvoNN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def get_model(self, model_name, train):
        self.online_network = torch.load(f"{self.model_directory}/{model_name}_o.pt", weights_only=False)
        self.target_network = torch.load(f"{self.model_directory}/{model_name}_t.pt", weights_only=False)
        if(train): self.online_network.train()
        else: self.online_network.eval()
        self.target_network.eval()
        return
    
    def save_model(self, model_name):
        torch.save(self.online_network, f"{self.model_directory}/{model_name}_o.pt")
        torch.save(self.target_network, f"{self.model_directory}/{model_name}_t.pt")
        print(f"Saved {model_name} successfully.")
        return
    
    def save_reward(self):
        np.savetxt(f"{self.model_directory}/reward_hist.txt", np.array(self.reward_hist), fmt="%s", delimiter=' ')
        return
    
    def add_reward(self, reward):
        self.reward_hist.append(reward)
        return
    
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done]) # s, a, r, s+1, d
        return

    @torch.no_grad
    def pick_action(self, state):
        output = self.online_network(state)
        optimal_action = torch.argmax(output[0]).item()
        random_action = int(np.random.randint(0, self.output_dim))
        action = np.random.choice([optimal_action, random_action], p=[1.0-self.epsilon, self.epsilon])
        return action
    
    def replay(self):
        if(len(self.memory)<self.batch_size): return
        batch = np.array(random.sample(self.memory, self.batch_size), dtype=object)
        state_pt = torch.from_numpy(np.stack(batch[:, 0])).float()
        action_pt = torch.from_numpy(np.stack(batch[:, 1])).long()
        reward_pt = torch.from_numpy(np.stack(batch[:, 2])).float()
        next_state_pt = torch.from_numpy(np.stack(batch[:, 3])).float()
        done_pt = torch.from_numpy(np.stack(batch[:, 4])).long()
        with torch.no_grad():
            current_q = self.target_network(state_pt)
            target_q = reward_pt+self.discount*torch.max(self.target_network(next_state_pt), dim=1)[0]*(1-done_pt)
            current_q[torch.arange(self.batch_size), action_pt] = target_q
        self.online_network.fit(state_pt, current_q, self.loss_func, self.optimizer)

    def update_values(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        if(self.time == 0): self.target_network.load_state_dict(self.online_network.state_dict())
        self.time = (self.time+1)%self.target_net_update_int

class DoubleDQN(DQN):
    
    def __init__(self, input_dim, output_dim, model_name="double_dqn"):
        super().__init__(input_dim, output_dim, model_name)

    def replay(self):
        if(len(self.memory)<self.batch_size): return
        batch = np.array(random.sample(self.memory, self.batch_size), dtype=object)
        state_pt = torch.from_numpy(np.stack(batch[:, 0])).float()
        action_pt = torch.from_numpy(np.stack(batch[:, 1])).long()
        reward_pt = torch.from_numpy(np.stack(batch[:, 2])).float()
        next_state_pt = torch.from_numpy(np.stack(batch[:, 3])).float()
        done_pt = torch.from_numpy(np.stack(batch[:, 4])).long()
        with torch.no_grad():
            next_state_best_action = torch.argmax(self.online_network(next_state_pt), dim=1, keepdim=True)
            current_q = self.target_network(state_pt)
            next_best_target_q = self.target_network(next_state_pt).gather(1, next_state_best_action).squeeze(1)
            target_q = reward_pt+self.discount*next_best_target_q*(1-done_pt)
            current_q[torch.arange(self.batch_size), action_pt] = target_q
        self.online_network.fit(state_pt, current_q, self.loss_func, self.optimizer)

    def update_values(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        self.target_network.soft_update(self.online_network)

class DuelingDQN(DQN):
    
    def __init__(self, input_dim, output_dim, model_name="dueling_dqn"):
        super().__init__(input_dim, output_dim, model_name)
        self.online_network = DuelingNetWork(input_dim, output_dim)
        self.target_network = DuelingNetWork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)

    def update_values(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        self.target_network.soft_update(self.online_network)

class DuelingDoubleDQN(DQN):
     
    def __init__(self, input_dim, output_dim, model_name="double_ddqn"):
        super().__init__(input_dim, output_dim, model_name)
        self.online_network = DuelingNetWork(input_dim, output_dim)
        self.target_network = DuelingNetWork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
    
    def replay(self):
        if(len(self.memory)<self.batch_size): return
        batch = np.array(random.sample(self.memory, self.batch_size), dtype=object)
        state_pt = torch.from_numpy(np.stack(batch[:, 0])).float()
        action_pt = torch.from_numpy(np.stack(batch[:, 1])).long()
        reward_pt = torch.from_numpy(np.stack(batch[:, 2])).float()
        next_state_pt = torch.from_numpy(np.stack(batch[:, 3])).float()
        done_pt = torch.from_numpy(np.stack(batch[:, 4])).long()
        with torch.no_grad():
            next_state_best_action = torch.argmax(self.online_network(next_state_pt), dim=1, keepdim=True)
            current_q = self.target_network(state_pt)
            next_best_target_q = self.target_network(next_state_pt).gather(1, next_state_best_action).squeeze(1)
            target_q = reward_pt+self.discount*next_best_target_q*(1-done_pt)
            current_q[torch.arange(self.batch_size), action_pt] = target_q
        self.online_network.fit(state_pt, current_q, self.loss_func, self.optimizer)

    def update_values(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        self.target_network.soft_update(self.online_network)

    
