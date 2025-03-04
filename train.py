import pygame
import torch
from environment import Game

from model.dqn_variant import DuelingDoubleDQN
from model.ac_variant import A2C

pygame.init()

env = Game()
input_dim = env.INPUT_SHAPE
output_dim = env.OUTPUT_SHAPE

def train_dqn(
    noisy=True,
    checkpoint=2000
):
    agent = DuelingDoubleDQN(input_dim, output_dim, noisy=noisy)
    agent.set_value(learning_rate=0.001)

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


def train_a2c(checkpoint=2000):
    agent = A2C(input_dim, output_dim)

    for i in range(agent.num_episode):
        state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
        total_reward = 0
        while(True):
            action_probability, value = agent.act(state.unsqueeze(0))
            action_distribution = torch.distributions.Categorical(action_probability)
            action = action_distribution.sample()

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
            total_reward+=reward

            agent.add_ep_values(reward, value, action_distribution.log_prob(action).unsqueeze(0))
            state = next_state
            if(done == True): break

        agent.fit()
        agent.clear_ep_values()

        print(f"ep {i+1}: {total_reward}")
        agent.add_reward(total_reward)
        env.reset()
        
        if((i+1)%checkpoint == 0):
            agent.save_model(f"snake_ep_{i+1}")
            agent.save_reward()

