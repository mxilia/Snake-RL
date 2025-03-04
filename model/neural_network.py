import torch
import torch.nn as nn
import torch.nn.functional as F

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
