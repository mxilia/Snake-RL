import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    # Structure: input (grid) -> 64 -> 64 -> 4 Q-Value for wasd
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def fit(self, x, y_true, optimizer, loss_func):
        optimizer.zero_grad()
        y_pred = self(x)
        loss = loss_func(y_pred, y_true)
        loss.backward()
        print(loss.item())
        optimizer.step()
        return
    
    def soft_update(self, target_net, tau=0.005):
        for target_param, main_param in zip(target_net.parameters(), self.parameters()):
            main_param.data.copy_(tau*target_param.data+(1.0-tau)*main_param.data)
