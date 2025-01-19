import pygame
from player import Player

pygame.init()

SCR_WIDTH = 800
SCR_HEIGHT = 600
run = True

screen = pygame.display.set_mode((SCR_WIDTH, SCR_HEIGHT))

plr = Player()

def checkEvent():
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            global run 
            run = False
    return

def paint():
    plr.draw(screen)
    pygame.display.update()
    return

def update():
    checkEvent()
    key = pygame.key.get_pressed()
    return

if __name__ == "__main__":
    while run:
        update()
        paint()
    pygame.quit()