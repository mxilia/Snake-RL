import pygame
import torch
from environment import Game
from agent import DoubleDQN as Agent

pygame.init()

clock = pygame.time.Clock()
env = Game()
input_dim = env.INPUT_SHAPE
output_dim = env.OUTPUT_SHAPE

agent = Agent(input_dim, output_dim)

agent.get_model("snake_ep_90000", False)
agent.epsilon = 0.0
state = torch.tensor(env.get_state()).reshape(input_dim)

while(True):
    action = agent.pick_action(state)
    next_state, total_reward, done = env.step(action, fps=60)
    next_state = torch.tensor(next_state).reshape(input_dim)
    state = next_state
    if(done == True): break
pygame.quit()