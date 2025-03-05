import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utility as util

checkpoint_path = "./checkpoints"

class Actor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.get_conv_out_dim(input_dim), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

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
        x = torch.softmax(self.fc4(x), dim=-1)
        return x
    
    def fit(self, log_prob, advantage, optimizer):
        optimizer.zero_grad()
        loss = -(log_prob*advantage).mean()
        loss.backward()
        optimizer.step()
        return loss.item()
    
class Critic(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.get_conv_out_dim(input_dim), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

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
    
    def fit(self, value, returns, loss_func, optimizer):
        optimizer.zero_grad()
        loss = loss_func(value, returns)
        loss.backward()
        optimizer.step()
        return loss.item()

class A2C:

    def __init__(self, input_dim, output_dim, model_name="a2c"):
        self.model_name = model_name
        self.model_directory = f"{checkpoint_path}/{model_name}"
        util.create_directory(self.model_directory)

        self.num_episode = 100000
        self.discount = 0.99
        self.lr_a = 0.00002
        self.lr_c = 0.00002

        self.value_ep = []
        self.reward_ep = []
        self.log_prob_ep = []
        self.reward_hist = []

        self.actor_network = Actor(input_dim, output_dim)
        self.critic_network = Critic(input_dim)
        self.optimizer_a = optim.Adam(self.actor_network.parameters(), lr=self.lr_a)
        self.optimizer_c = optim.Adam(self.critic_network.parameters(), lr=self.lr_c)
        self.loss_func_c = nn.MSELoss()

    def set_value(self,
        num_episode = 100000,
        discount = 0.99,
        lr_a = 0.00002,
        lr_c = 0.00002
    ):
        self.num_episode = num_episode
        self.discount = discount
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.optimizer_a = optim.Adam(self.actor_network.parameters(), lr=self.lr_a)
        self.optimizer_c = optim.Adam(self.critic_network.parameters(), lr=self.lr_c)
        return

    def get_model(self, episode, train):
        self.actor_network.load_state_dict(torch.load(f"{self.model_directory}/ep_{episode}_a.pt"))
        self.critic_network.load_state_dict(torch.load(f"{self.model_directory}/ep_{episode}_c.pt"))
        if(train == False): 
            self.actor_network.eval()
            self.critic_network.eval()
        return
    
    def save_model(self, episode):
        torch.save(self.actor_network.state_dict(), f"{self.model_directory}/ep_{episode}_a.pt")
        torch.save(self.critic_network.state_dict(), f"{self.model_directory}/ep_{episode}_c.pt")
        print(f"Saved {self.model_name} (ep_{episode}) successfully.")
        return

    def save_reward(self):
        np.savetxt(f"{self.model_directory}/reward_hist.txt", np.array(self.reward_hist), fmt="%s", delimiter=' ')
        return
    
    def add_reward(self, reward):
        self.reward_hist.append(reward)
        return
    
    def act(self, x):
        return self.actor_network(x), self.critic_network(x)
    
    def add_ep_values(self, reward, value, log_prob):
        self.reward_ep.append(reward)
        self.value_ep.append(value)
        self.log_prob_ep.append(log_prob)
        return

    def clear_ep_values(self):
        self.value_ep.clear()
        self.reward_ep.clear()
        self.log_prob_ep.clear()
        return

    def fit(self):
        returns = []
        for t in range(0, len(self.reward_ep)-1): returns.append(self.reward_ep[t]+self.discount*self.value_ep[t+1])
        returns.append(self.reward_ep[-1])
        returns = torch.tensor(returns, dtype=torch.float32)
        value_ep = torch.cat(self.value_ep).squeeze()
        advantage = returns-value_ep.detach()
        log_prob_ep = torch.cat(self.log_prob_ep)
        self.actor_network.fit(log_prob_ep, advantage, self.optimizer_a)
        self.critic_network.fit(value_ep, returns, self.loss_func_c, self.optimizer_c)
        return