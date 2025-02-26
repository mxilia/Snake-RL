import pygame
from environment import Game

pygame.init()

clock = pygame.time.Clock()
env = Game()

while(True):
    env.check_event()
    env.update()
    env.draw()
    if(env.plr.alive == False): break
    clock.tick(10)
pygame.quit()