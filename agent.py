import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random

import utility as util 
from neural_network import ConvoNN
from neural_network import DuelingNetWork

checkpoint_path = "./checkpoints"

class DQN:
    
    def __init__(self, input_dim, output_dim, noisy=False, soft_update=True, model_name="normal_dqn"):
        self.model_directory = f"{checkpoint_path}/{"noisy_" if(noisy == True) else ""}{model_name}"
        util.create_directory(self.model_directory)

        self.noisy = noisy
        self.soft_update = soft_update

        self.num_episode = 10000

        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.02

        self.discount = 0.99
        self.learning_rate = 0.0001

        self.batch_size = 32
        self.memory_size = 200000
        self.target_net_update_int = 500
        self.time = 0

        self.memory = deque(maxlen=self.memory_size)
        self.reward_hist = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.online_network = ConvoNN(input_dim, output_dim, noisy=self.noisy)
        self.target_network = ConvoNN(input_dim, output_dim, noisy=self.noisy)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def set_value(self,
        num_episode=10000,
        epsilon = 1.0,
        epsilon_decay = 0.99999,
        epsilon_min = 0.02,
        discount = 0.99,
        learning_rate = 0.0001,
        batch_size = 32,
        memory_size = 200000,
        target_net_update_int = 500
    ):
        self.num_episode = num_episode
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount = discount
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_net_update_int = target_net_update_int
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        return

    def get_model(self, model_name, train):
        self.online_network.load_state_dict(torch.load(f"{self.model_directory}/{model_name}_o.pt"))
        self.target_network.load_state_dict(torch.load(f"{self.model_directory}/{model_name}_t.pt"))
        if(train == False): self.online_network.eval()
        return
    
    def save_model(self, model_name):
        torch.save(self.online_network.state_dict(), f"{self.model_directory}/{model_name}_o.pt")
        torch.save(self.target_network.state_dict(), f"{self.model_directory}/{model_name}_t.pt")
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
        if(self.noisy == True): return optimal_action
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
        if(self.noisy == True):
            self.online_network.reset_noise()
            self.target_network.reset_noise()

    def update_values(self):
        if(self.noisy == False): self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        if(self.soft_update): self.target_network.soft_update(self.online_network)
        else:
            if(self.time == 0): self.target_network.load_state_dict(self.online_network.state_dict())
            self.time = (self.time+1)%self.target_net_update_int

class DoubleDQN(DQN):
    
    def __init__(self, input_dim, output_dim, noisy=False, soft_update=True, model_name="double_dqn"):
        super().__init__(input_dim, output_dim, noisy, soft_update, model_name)

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
        if(self.noisy == True):
            self.online_network.reset_noise()
            self.target_network.reset_noise()

class DuelingDQN(DQN):
    
    def __init__(self, input_dim, output_dim, noisy=False, soft_update=True, model_name="dueling_dqn"):
        super().__init__(input_dim, output_dim, noisy, soft_update, model_name)
        self.online_network = DuelingNetWork(input_dim, output_dim, noisy=self.noisy)
        self.target_network = DuelingNetWork(input_dim, output_dim, noisy=self.noisy)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)

class DuelingDoubleDQN(DQN):
     
    def __init__(self, input_dim, output_dim, noisy=False, soft_update=True, model_name="dueling_ddqn"):
        super().__init__(input_dim, output_dim, noisy, soft_update, model_name)
        self.online_network = DuelingNetWork(input_dim, output_dim, noisy=self.noisy)
        self.target_network = DuelingNetWork(input_dim, output_dim, noisy=self.noisy)
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
        if(self.noisy == True):
            self.online_network.reset_noise()
            self.target_network.reset_noise()