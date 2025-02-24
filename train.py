import pygame
import torch
from environment import Game
from agent import DQN

pygame.init()

env = Game()
input_dim = env.INPUT_SHAPE
output_dim = env.OUTPUT_SHAPE

agent = DQN(input_dim, output_dim, "normal_dqn")
checkpoint = 10000

for i in range(agent.num_episode):
    state = torch.tensor(env.get_state()).reshape(input_dim)
    while(True):
        action = agent.pick_action(state)
        next_state, total_reward, done = env.step(action)
        next_state = torch.tensor(next_state).reshape(input_dim)

        agent.add_memory(state, action, total_reward, next_state, done)
        agent.replay()
        agent.update_values()

        state = next_state
        if(done == True): break

    print(f"ep {i+1}: {env.get_reward()}")
    agent.add_reward(env.get_reward())
    env.reset()
    
    if((i+1)%checkpoint == 0):
        agent.save_model(f"snake_ep_{i+1}")
        agent.save_reward()


