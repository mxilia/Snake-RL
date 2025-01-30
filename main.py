import pygame
import numpy as np
from environment import Environment
from ai import Neural_Net

pygame.init()
clock = pygame.time.Clock()

user = False
run = True
env = Environment()
bot = Neural_Net(env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL)

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
        if(user == False):
            bot.pick_action(np.array(env.getState()).reshape(1, bot.input_node))
        update()
        paint()
        clock.tick(120)
    return

def train():
    for i in range(bot.episode):
        play()
        bot.reset()
        env.reset()
        bot.back_prop()
    return

if __name__ == "__main__":
    if(user == False): train()
    else: play()
    pygame.quit()