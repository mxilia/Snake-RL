import pygame
from environment import Game

pygame.init()

clock = pygame.time.Clock()
env = Game()

env.set_display(2)

while(True):
    if(env.plr.alive == False): break
    env.check_event()
    env.update()
    env.draw()
    clock.tick(30)
pygame.quit()