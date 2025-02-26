import torch
import torch.nn as nn

class NeuralNetWork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
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
    
    def soft_update(self, goal_net, tau=0.005):
        for self_param, goal_param in zip(self.parameters(), goal_net.parameters()):
            self_param.data.copy_((1.0-tau)*self_param.data+tau*goal_param.data)

class DuelingNetWork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_value = nn.Linear(64, 32)
        self.fc_advantage = nn.Linear(64, 32)
        self.value = nn.Linear(32, 1)
        self.advantage = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        V = torch.relu(self.fc_value(x))
        A = torch.relu(self.fc_advantage(x))
        V = self.value(V)
        A = self.advantage(A)
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
