import pygame
import numpy as np
from environment import Environment
from agent import DQN

pygame.init()
clock = pygame.time.Clock()

user = False
train = True
run = True
env = Environment()
agent = DQN(env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL)

key_W = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
key_A = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
key_S = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
key_D = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
keys = [key_W, key_A, key_S, key_D]

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
        if(user == False and env.plr.completeMovement()):
            if(agent.done == True): break
            action = agent.pick_action(np.array(env.getState()).reshape(env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL,))
            pygame.event.post(keys[action])
            agent.update_reward(env.plr.getSize(), not env.plr.alive)
            agent.record(action, np.array(env.emulate(action)).reshape(env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL,))
            agent.replay()
        update()
        paint()
        #clock.tick(120)
    agent.update_reward(env.plr.getSize(), not env.plr.alive)
    return

def train_agent():
    #agent.get_model()
    for i in range(agent.episode):
        agent.reset()
        env.reset()
        agent.setCurrentState(np.array(env.getState()).reshape(env.SCR_WIDTH_PIXEL*env.SCR_HEIGHT_PIXEL,))
        play()
    agent.save_model()
    return

if __name__ == "__main__":
    if(user == False): 
        if(train == True):
            train_agent()
        else: play()
    else: play()
    pygame.quit()
    if(train == True):
        agent.result()
        