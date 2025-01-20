import pygame
from player import Player
from apple import Apple
from mqueue import Queue

pygame.init()

SCR_WIDTH = 800
SCR_HEIGHT = 600
run = True

screen = pygame.display.set_mode((SCR_WIDTH, SCR_HEIGHT))
clock = pygame.time.Clock()

pygame_key = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]
key_order = Queue()

plr = Player(SCR_WIDTH, SCR_HEIGHT)
apple = Apple(SCR_WIDTH, SCR_HEIGHT)

def checkEvent():
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            global run 
            run = False
        elif e.type == pygame.KEYDOWN:
            if(e.key == pygame.K_w and e.key != pygame_key[plr.last_key]):
                key_order.push(0)
            if(e.key == pygame.K_a and e.key != pygame_key[plr.last_key]):
                key_order.push(1)
            if(e.key == pygame.K_s and e.key != pygame_key[plr.last_key]):
                key_order.push(2)
            if(e.key == pygame.K_d and e.key != pygame_key[plr.last_key]):
                key_order.push(3)
    return

def paint():
    screen.fill((0, 0, 0))
    plr.draw(screen)
    apple.draw(screen)
    pygame.display.update()
    return

def update():
    checkEvent()
    if(not key_order.empty() and plr.move(key_order.front(), True)):
        key_order.pop()
    else: 
        plr.move(plr.last_key, False)
    apple.generate()
    return

if __name__ == "__main__":
    while run:
        update()
        paint()
        clock.tick(60)
    pygame.quit()