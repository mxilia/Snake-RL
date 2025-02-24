import pygame
from environment import Game

pygame.init()

clock = pygame.time.Clock()
env = Game()

while(True):
    if(env.plr.complete_movement()): print(env.get_reward())
    if(env.plr.alive == False): break
    env.check_event()
    env.update()
    env.draw()
    clock.tick(60)
pygame.quit()