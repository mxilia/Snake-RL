import torch
import torch.nn as nn

from utility import NoisyLinear

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
    
    def soft_update(self, goal_net, tau=0.005):
        for self_param, goal_param in zip(self.parameters(), goal_net.parameters()):
            self_param.data.copy_((1.0-tau)*self_param.data+tau*goal_param.data)

class DuelingNetWork(nn.Module):

    def __init__(self, input_dim, output_dim, noisy=False):
        super().__init__()
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
    
    def soft_update(self, goal_net, tau=0.005):
        for self_param, goal_param in zip(self.parameters(), goal_net.parameters()):
            self_param.data.copy_((1.0-tau)*self_param.data+tau*goal_param.data)
