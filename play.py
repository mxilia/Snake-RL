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
    if(env.plr.complete_movement()):
        total_reward+=env.get_reward()
        print(total_reward)
    if(env.plr.alive == False): break
    clock.tick(20)
pygame.quit()