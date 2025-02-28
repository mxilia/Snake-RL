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

    def __init__(self, in_features, out_features, sigma_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1/float(in_features)
        lower_bound = -torch.sqrt(torch.tensor(k)).item()
        upper_bound = torch.sqrt(torch.tensor(k)).item()
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features).uniform_(lower_bound, upper_bound))
        self.mu_b = nn.Parameter(torch.empty(out_features).uniform_(lower_bound, upper_bound))
        self.sigma_w = nn.Parameter(torch.ones(out_features, in_features)*sigma_init)
        self.sigma_b = nn.Parameter(torch.ones(out_features)*sigma_init)

    def forward(self, x):
        epsilon_w = torch.randn_like(self.sigma_w)
        epsilon_b = torch.randn_like(self.sigma_b)
        noisy_w = self.mu_w+self.sigma_w*epsilon_w
        noisy_b = self.mu_b+self.sigma_b*epsilon_b
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
