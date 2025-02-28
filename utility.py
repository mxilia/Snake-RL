import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class Queue:

    def __init__(self):
        self.dq = deque([])
        self.sz = 0

    def push(self, x):
        self.dq.append(x)
        self.sz+=1

    def pop(self):
        if(self.sz==0): return
        self.dq.popleft()
        self.sz-=1

    def empty(self):
        if(self.sz==0): return True
        return False
    
    def front(self):
        if(self.sz==0): return -1
        x = self.dq.popleft()
        self.dq.appendleft(x)
        return x
    
    def rear(self):
        if(self.sz==0): return -1
        x = self.dq.pop()
        self.dq.append(x)
        return x
    
    def copy_queue(self, queue):
        while(not queue.empty()):
            self.push(queue.front())
            queue.pop()
        return
    
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
    
def calculate_dist(a, b):
    return np.sqrt(np.square(a[0]-b[0])+np.square(a[1]-b[1]))

def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Create {directory_name} successfully.")
    except FileExistsError:
        print(f"{directory_name} existed.")
        return
    except PermissionError:
        print(f"Creating {directory_name} denied.")
        return
    except Exception as e:
        print(f"Creating {directory_name} error.")
        return
