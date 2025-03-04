import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utility as util
from model.neural_network import *

class A2C:

    def __init__(self, input_dim, output_dim, model_name="a2c"):
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

    def get_model(self, model_name, train):
        self.actor_network.load_state_dict(torch.load(f"{self.model_directory}/{model_name}_a.pt"))
        self.critic_network.load_state_dict(torch.load(f"{self.model_directory}/{model_name}_c.pt"))
        if(train == False): 
            self.actor_network.eval()
            self.critic_network.eval()
        return
    
    def save_model(self, model_name):
        torch.save(self.actor_network.state_dict(), f"{self.model_directory}/{model_name}_a.pt")
        torch.save(self.critic_network.state_dict(), f"{self.model_directory}/{model_name}_c.pt")
        print(f"Saved {model_name} successfully.")
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