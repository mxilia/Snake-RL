import pygame
import torch
from environment import Game
from agent import DuelingDoubleDQN as Agent

pygame.init()

env = Game()
input_dim = env.INPUT_SHAPE
output_dim = env.OUTPUT_SHAPE

agent = Agent(input_dim, output_dim, noisy=True)
checkpoint = 2000

for i in range(agent.num_episode):
    state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
    total_reward = 0
    while(True):
        action = agent.pick_action(state.unsqueeze(0))
        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
        total_reward+=reward

        agent.add_memory(state, action, total_reward, next_state, done)
        agent.replay()
        agent.update_values()

        state = next_state
        if(done == True): break

    print(f"ep {i+1}: {total_reward}")
    agent.add_reward(total_reward)
    env.reset()
    
    if((i+1)%checkpoint == 0):
        agent.save_model(f"snake_ep_{i+1}")
        agent.save_reward()


