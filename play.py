import pygame
from environment import Game

pygame.init()

clock = pygame.time.Clock()
env = Game()
total_reward = 0

def play():
    while(True):
        env.check_event()
        env.update()
        env.draw()
        if(env.plr.alive == False): break
        clock.tick(45)
    pygame.quit()