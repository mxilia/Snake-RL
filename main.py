import pygame
import numpy as np
from environment import Environment
from agent import Neural_Net

pygame.init()
clock = pygame.time.Clock()

user = False
run = True
env = Environment()
agent = Neural_Net(env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL)

def checkEvent():
    for e in pygame.event.get():
        if(e.type == pygame.QUIT):
            global run
            run = False
        else:
            env.checkEvent(e)
    return

def paint():
    env.draw()
    pygame.display.update()
    return

def update():
    checkEvent()
    env.update()
    return

def play():
    while(env.plr.alive and run):
        if(user == False and env.plr.completeMovement()):
            agent.pick_action(env.getNextState())
        update()
        paint()
        clock.tick(120)
    return

def train():
    for i in range(agent.episode):
        agent.reset()
        env.reset()
        agent.setCurrentState(env.getState())
        play()
        agent.back_prop()
    return

if __name__ == "__main__":
    if(user == False): train()
    else: play()
    pygame.quit()