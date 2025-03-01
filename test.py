import pygame
import torch
from environment import Game
from agent import DuelingDoubleDQN as Agent

pygame.init()

clock = pygame.time.Clock()
env = Game()
input_dim = env.INPUT_SHAPE
output_dim = env.OUTPUT_SHAPE

agent = Agent(input_dim, output_dim, noisy=True)

agent.get_model("snake_ep_10000", False)
agent.epsilon = 0.0
state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)

while(True):
    action = agent.pick_action(state.unsqueeze(0))
    next_state, total_reward, done = env.step(action, fps=60)
    next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
    state = next_state
    if(done == True): break
pygame.quit()