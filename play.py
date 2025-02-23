import pygame
from environment import Game

pygame.init()

clock = pygame.time.Clock()
env = Game()

while(True):
    if(env.plr.alive == False): break
    env.check_event()
    env.update()
    env.draw()
    clock.tick(10)
pygame.quit()