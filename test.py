import pygame
from environment import Game
from agent import Agent

pygame.init()

clock = pygame.time.Clock()
env = Game()
agent = Agent(env)

env.set_display(2)
agent.get_model("snake_ep_18000", False)
agent.set_current_state(env.get_state())
agent.epsilon = 0.0

while(True):
    if(env.plr.alive == False): break
    if(env.plr.complete_movement()):
        action = agent.pick_action(env.get_state())
        env.post_action(action)
    env.check_event()
    env.update()
    env.draw()
    clock.tick(60)
    
pygame.quit()