import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetWork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
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

env = gym.make("LunarLander-v2")
input_dim = 8
output_dim = 4

num_episode = 1000

epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.02

discount = 0.99
learning_rate = 0.001

batch_size = 32
memory_size = 100000
target_net_update_int = 500
time = 0

memory = deque(maxlen=memory_size)
reward_hist = []

online_network = NeuralNetWork(input_dim, output_dim)
target_network = NeuralNetWork(input_dim, output_dim)
optimizer = optim.Adam(online_network.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

for i in range(num_episode):
    state = torch.tensor(env.reset()[0])
    reward_ep = 0
    while(True):
        optimal_action = torch.argmax(online_network(state)).item()
        random_action = int(np.random.randint(0, output_dim))
        action = np.random.choice([optimal_action, random_action], p=[1.0-epsilon, epsilon])

        next_state,reward,done,_,__ = env.step(action)
        next_state = torch.tensor(next_state)

        reward_ep+=reward
        memory.append([state, action, reward, next_state, done]) # s, a, r, s+1, d

        if(len(memory)>=batch_size):
            batch = np.array(random.sample(memory, batch_size), dtype=object)
            state_pt = torch.from_numpy(np.stack(batch[:, 0])).float()
            action_pt = torch.from_numpy(np.stack(batch[:, 1])).long()
            reward_pt = torch.from_numpy(np.stack(batch[:, 2])).float()
            next_state_pt = torch.from_numpy(np.stack(batch[:, 3])).float()
            done_pt = torch.from_numpy(np.stack(batch[:, 4])).long()
            
            with torch.no_grad():
                next_state_best_action = torch.argmax(online_network(next_state_pt), dim=1, keepdim=True)
                current_q = target_network(state_pt)
                next_best_target_q = target_network(next_state_pt).gather(1, next_state_best_action).squeeze(1)
                target_q = reward_pt+discount*next_best_target_q*(1-done_pt)
                current_q[torch.arange(batch_size), action_pt] = target_q
            online_network.fit(state_pt, current_q, loss_func, optimizer)

        state = next_state
        epsilon = max(epsilon_min, epsilon*epsilon_decay)

        if(time == 0): target_network.load_state_dict(online_network.state_dict())
        time = (time+1)%target_net_update_int

        if(done): break

    print(f"ep {i+1}: {reward_ep}")
    reward_hist.append(reward_ep)

# Plot
plt.figure(figsize=(6,4), dpi=100)
plt.plot(reward_hist)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Test
env = gym.make("LunarLander-v2", render_mode="human")
state = torch.tensor(env.reset()[0])
while True:
    optimal_action = torch.argmax(online_network(state)).item()
    next_state,reward,done,_,__ = env.step(optimal_action)
    state = torch.tensor(next_state)
    env.render()
    if(done): break