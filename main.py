import pygame
from player import Player

pygame.init()

SCR_WIDTH = 800
SCR_HEIGHT = 600
run = True

screen = pygame.display.set_mode((SCR_WIDTH, SCR_HEIGHT))
clock = pygame.time.Clock()

plr = Player(SCR_WIDTH, SCR_HEIGHT)

def checkEvent():
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            global run 
            run = False
    return

def paint():
    screen.fill((0, 0, 0))
    plr.draw(screen)
    pygame.display.update()
    return

def update():
    checkEvent()
    key = pygame.key.get_pressed()
    if key[pygame.K_w]:
        plr.move(0, -plr.speed)
    elif key[pygame.K_a]:
        plr.move(-plr.speed, 0)
    elif key[pygame.K_s]:
        plr.move(0, plr.speed)
    elif key[pygame.K_d]:
        plr.move(plr.speed, 0)
    return

if __name__ == "__main__":
    while run:
        update()
        paint()
        clock.tick(60)
    pygame.quit()