import pygame
from environment import Game
from agent import Agent

pygame.init()

env = Game()
agent = Agent(env)

checkpoint = 2000

for i in range(agent.episode):
    print(f"Episode: {i}")
    agent.set_current_state(env.get_state())

    while(True):
        if(env.plr.alive == False): break
        if(env.plr.complete_movement()):
            action = agent.pick_action(env.get_state())
            agent.record(action, env.emulate(action))
            agent.replay()
            env.post_action(action)
        env.check_event()
        env.update()
        env.draw()

    agent.add_reward()
    env.reset()
    if((i+1)%checkpoint == 0):
        agent.save_model(f"snake_ep_{i+1}")
        agent.save_reward()


