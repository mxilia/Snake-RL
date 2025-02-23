import pygame
import torch
from environment import Game
from agent import DQN

pygame.init()

clock = pygame.time.Clock()
env = Game()
input_dim = env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL
output_dim = len(env.keys)

agent = DQN(input_dim, output_dim, model_name="normal_dqn")

agent.get_model("snake_ep_100000", False)
agent.epsilon = 0.0

while(True):
    if(env.plr.alive == False): break
    if(env.plr.complete_movement()):
        action = agent.pick_action(torch.tensor(env.get_state()).reshape(input_dim))
        env.post_action(action)
    env.check_event()
    env.update()
    env.draw()
    clock.tick(60)
    
pygame.quit()