import pygame
from environment import Game

pygame.init()

clock = pygame.time.Clock()
env = Game()
total_reward = 0

while(True):
    env.check_event()
    env.update()
    env.draw()
    env.get_frames()
    if(env.plr.alive == False): break
    clock.tick(60)
pygame.quit()