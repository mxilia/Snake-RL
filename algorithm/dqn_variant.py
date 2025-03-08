import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random

import utility as util

checkpoint_path = "./checkpoints"

class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        self.register_buffer('epsilon_w', torch.FloatTensor(self.out_features, self.in_features))
        self.register_buffer('epsilon_b', torch.FloatTensor(self.out_features))
        self.reset_param()
        self.reset_noise()

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign()*x.abs().sqrt()
    
    def reset_param(self):
        k = 1/torch.sqrt(torch.tensor(self.in_features, dtype=torch.float32))
        self.mu_w.data.uniform_(-k, k)
        self.mu_b.data.uniform_(-k, k)
        self.sigma_w.data.fill_(value=self.sigma_init/torch.sqrt(torch.tensor(self.in_features, dtype=torch.float32)))
        self.sigma_b.data.fill_(value=self.sigma_init/torch.sqrt(torch.tensor(self.in_features, dtype=torch.float32)))
        return
    
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.epsilon_w.copy_(torch.outer(epsilon_out, epsilon_in))
        self.epsilon_b.copy_(epsilon_out)
        return
    
    def forward(self, x):
        if(self.training == True):
            noisy_w = self.mu_w+self.sigma_w*self.epsilon_w
            noisy_b = self.mu_b+self.sigma_b*self.epsilon_b
        else:
            noisy_w = self.mu_w
            noisy_b = self.mu_b
        return F.linear(x, noisy_w, noisy_b)

class ConvoNN(nn.Module):

    def __init__(self, input_dim, output_dim, noisy=False):
        super().__init__()
        self.noisy = noisy
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        if(noisy == True): self.Linear = NoisyLinear
        else: self.Linear = nn.Linear
        self.fc1 = self.Linear(self.get_conv_out_dim(input_dim), 128)
        self.fc2 = self.Linear(128, 128)
        self.fc3 = self.Linear(128, 128)
        self.fc4 = self.Linear(128, output_dim)

    @torch.no_grad
    def get_conv_out_dim(self, input_dim):
        x = torch.zeros(input_dim)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.flatten(x).shape[0]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def fit(self, x, y_true, loss_func, optimizer):
        optimizer.zero_grad()
        y_pred = self(x)
        loss = loss_func(y_pred, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad
    def soft_update(self, goal_net, tau=0.005):
        for self_param, goal_param in zip(self.parameters(), goal_net.parameters()):
            self_param.data.copy_((1.0-tau)*self_param.data+tau*goal_param.data)
    
    def reset_noise(self):
        if(self.noisy == False): return
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
        return

class DuelingNetWork(nn.Module):

    def __init__(self, input_dim, output_dim, noisy=False):
        super().__init__()
        self.noisy = noisy
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        if(noisy == True): self.Linear = NoisyLinear
        else: self.Linear = nn.Linear
        self.fc1 = self.Linear(self.get_conv_out_dim(input_dim), 128)
        self.fc2 = self.Linear(128, 64)
        self.fc3 = self.Linear(64, 64)
        self.value = self.Linear(64, 1)
        self.advantage = self.Linear(64, output_dim)

    @torch.no_grad
    def get_conv_out_dim(self, input_dim):
        x = torch.zeros(input_dim)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return torch.flatten(x).shape[0]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        V = self.value(x)
        A = self.advantage(x)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)
        return Q
    
    def fit(self, x, y_true, loss_func, optimizer):
        optimizer.zero_grad()
        y_pred = self(x)
        loss = loss_func(y_pred, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad
    def soft_update(self, goal_net, tau=0.005):
        for self_param, goal_param in zip(self.parameters(), goal_net.parameters()):
            self_param.data.copy_((1.0-tau)*self_param.data+tau*goal_param.data)

    def reset_noise(self):
        if(self.noisy == False): return
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()

class DQN:
    
    def __init__(self, input_dim, output_dim, noisy=False, dueling=False, soft_update=True, model_name="normal_dqn"):
        self.model_name = model_name
        self.model_directory = f"{checkpoint_path}/{model_name}"
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
        self.tau = 0.005
        self.time = 0

        self.buffer = deque(maxlen=self.memory_size)
        self.reward_hist = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if(dueling == True): self.structure = DuelingNetWork
        else: self.structure = ConvoNN
        self.online_network = self.structure(input_dim, output_dim, noisy=self.noisy)
        self.target_network = self.structure(input_dim, output_dim, noisy=self.noisy)
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
        target_net_update_int = 500,
        tau = 0.005
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
        self.tau = tau
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        return
    
    def get_directory(self):
        return self.model_directory

    def get_model(self, episode, train):
        self.online_network.load_state_dict(torch.load(f"{self.model_directory}/{episode}_o.pt"))
        self.target_network.load_state_dict(torch.load(f"{self.model_directory}/{episode}_t.pt"))
        if(train == False): self.online_network.eval()
        return
    
    def save_model(self, episode):
        torch.save(self.online_network.state_dict(), f"{self.model_directory}/{episode}_o.pt")
        torch.save(self.target_network.state_dict(), f"{self.model_directory}/{episode}_t.pt")
        print(f"Saved {self.model_name} ({episode}) successfully.")
        return
    
    def save_reward(self):
        np.savetxt(f"{self.model_directory}/reward_hist.txt", np.array(self.reward_hist), fmt="%s", delimiter=' ')
        return
    
    def add_reward(self, reward):
        self.reward_hist.append(reward)
        return
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done]) # s, a, r, s+1, d
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
        if(len(self.buffer)<self.batch_size): return
        batch = np.array(random.sample(self.buffer, self.batch_size), dtype=object)
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
        if(self.noisy == True): self.online_network.reset_noise()

    def update_values(self):
        if(self.noisy == False): self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        if(self.soft_update): self.target_network.soft_update(self.online_network, self.tau)
        else:
            if(self.time == 0): self.target_network.load_state_dict(self.online_network.state_dict())
            self.time = (self.time+1)%self.target_net_update_int

class DoubleDQN(DQN):
    
    def __init__(self, input_dim, output_dim, noisy=False, dueling=False, soft_update=True, model_name="double_dqn"):
        super().__init__(input_dim, output_dim, noisy, dueling, soft_update, model_name)

    def replay(self):
        if(len(self.buffer)<self.batch_size): return
        batch = np.array(random.sample(self.buffer, self.batch_size), dtype=object)
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
        if(self.noisy == True): self.online_network.reset_noise()