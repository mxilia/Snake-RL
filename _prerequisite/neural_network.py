import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class NeuralNetWork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
    def fit(self, x, y_true, loss_func, optimizer):
        optimizer.zero_grad()
        y_pred = self(x)
        loss = loss_func(y_pred, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()

agent = NeuralNetWork(2, 1)
loss_func = nn.MSELoss()
optimizer = optim.Adam(agent.parameters(), lr=0.1)

x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
loss_hist = []

def train(epoch=500):
    for i in range(epoch):
        loss = agent.fit(x, y, loss_func, optimizer)
        print(f"Loss: {loss}")
        loss_hist.append(loss)
    return

def test():
    prediction = agent(x)
    print(prediction)
    return

def plot_loss():
    plt.figure(figsize=(6,4), dpi=100)
    plt.plot(loss_hist)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    return

train()
test()
plot_loss()