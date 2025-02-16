import pygame
import numpy as np
from environment import Snake_Game
from agent import DQN

pygame.init()
clock = pygame.time.Clock()

run = True
env = Snake_Game()

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
    while(True):
        if(not run): break
        if(env.plr.alive == False): break
        update()
        paint()
        clock.tick(120)
    return

play()
pygame.quit()