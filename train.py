import pygame
import torch
from environment import Game
from agent import DQN

pygame.init()

env = Game()
input_dim = env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL
output_dim = len(env.keys)

agent = DQN(input_dim, output_dim, "normal_dqn")
checkpoint = 2000

for i in range(agent.num_episode):
    state = torch.tensor(env.get_state()).reshape(input_dim)
    while(True):
        if(env.plr.alive == False): break
        if(env.plr.complete_movement()):
            action = agent.pick_action(state)
            env.post_action(action)
            next_state = torch.tensor(env.emulate(action)).reshape(input_dim)

            agent.add_memory(state, action, env.get_reward(), next_state, not env.plr.alive)
            agent.replay()
            agent.update_values()

            state = next_state

        env.check_event()
        env.update()
        env.draw()

    print(f"ep {i+1}: {env.get_reward()}")
    agent.add_reward(env.get_reward())
    env.reset()
    
    if((i+1)%checkpoint == 0):
        agent.save_model(f"snake_ep_{i+1}")
        agent.save_reward()


