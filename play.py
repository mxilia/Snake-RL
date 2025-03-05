import pygame

clock = pygame.time.Clock()

def play(env):
    while(True):
        env.check_event()
        env.update()
        env.draw()
        if(env.plr.alive == False): break
        clock.tick(45)
    pygame.quit()