import pygame
import numpy as np
from player import Player
from apple import Apple
from mqueue import Queue
from ai import Neural_Net

pygame.init()

user = True
pixel_size = 20

SCR_WIDTH = 800
SCR_HEIGHT = 600
SCR_WIDTH_PIXEL = int(SCR_WIDTH/pixel_size)
SCR_HEIGHT_PIXEL = int(SCR_HEIGHT/pixel_size)

run = True
screen = pygame.display.set_mode((SCR_WIDTH, SCR_HEIGHT))
clock = pygame.time.Clock()

key_order = Queue()
bot = Neural_Net(SCR_WIDTH_PIXEL*SCR_HEIGHT_PIXEL)
plr = Player(SCR_WIDTH, SCR_HEIGHT)
apple = Apple(SCR_WIDTH, SCR_HEIGHT)

pixel_grid = []

def checkEvent():
    if(user == True):
        for e in pygame.event.get():
            if(e.type == pygame.QUIT):
                global run 
                run = False
            elif(e.type == pygame.KEYDOWN):
                current_key = -1
                if(key_order.empty): current_key = plr.rect[0][0]
                else: current_key = key_order.rear()
                if(e.key == pygame.K_w and current_key%2 != 0):
                    key_order.push(0)
                if(e.key == pygame.K_a and current_key%2 != 1):
                    key_order.push(1)
                if(e.key == pygame.K_s and current_key%2 != 0):
                    key_order.push(2)
                if(e.key == pygame.K_d and current_key%2 != 1):
                    key_order.push(3)
    else:
        current_key = -1
        if(key_order.empty): current_key = plr.rect[0][0]
        else: current_key = key_order.rear()
        if(bot.decision == 0 and current_key%2 != 0):
            key_order.push(0)
        if(bot.decision == 1 and current_key%2 != 1):
            key_order.push(1)
        if(bot.decision == 2 and current_key%2 != 0):
            key_order.push(2)
        if(bot.decision == 3 and current_key%2 != 1):
            key_order.push(3)
        bot.move -= 1
    return

def checkStatus():
    global run
    if(plr.alive == False):
        run = False

def paint():
    screen.fill((0, 0, 0))
    plr.draw(screen)
    apple.draw(screen)
    pygame.display.update()
    return

def update():
    checkEvent()
    if(not key_order.empty() and plr.changeDir(key_order.front())):
        key_order.pop()
    plr.move()
    plr.grow(apple.collide(plr.rect[0][1].x, plr.rect[0][1].y))
    apple.generate(plr.rect)
    checkStatus()
    return

def update_pixel_grid():
    pixel_grid.clear()
    for i in range(SCR_HEIGHT_PIXEL):
        row = []
        for j in range(SCR_WIDTH_PIXEL):
            row.append(0.0)
        pixel_grid.append(row)
    pixel_grid[apple.getPixelY()][apple.getPixelX()] = 1.0
    for e in plr.getBodyPixel():
        pixel_grid[e[1]][e[0]] = 1.0
    return

def criticize_bot():
    list = [0.0, 0.0, 0.0, 0.0]
    dist_x = apple.getPixelX()-plr.getPixelX(0)
    dist_y = apple.getPixelY()-plr.getPixelY(0)
    if(dist_y>0 and plr.getPixelY(0)-1>=0 and not plr.collideBody(0, -1)): list[2] = 1.0
    elif(dist_y<0 and plr.getPixelY(0)+1<SCR_HEIGHT_PIXEL and not plr.collideBody(0, 1)):  list[0] = 1.0
    if(dist_x>0 and plr.getPixelX(0)-1>=0 and not plr.collideBody(-1, 0)): list[3] = 1.0
    elif(dist_x<0 and plr.getPixelX(0)+1<SCR_WIDTH_PIXEL and not plr.collideBody(1, 0)): list[1] = 1.0
    else: 
        list[0] = 1.0
        list[1] = 1.0
        list[2] = 1.0
        list[3] = 1.0
    return np.array(list).reshape(1, len(list))

def bot_train():
    global run
    for i in range(50):
        while(run):
            bot.predict(np.array(pixel_grid).reshape(1, bot.input_node), criticize_bot())
            update()
            paint()
            update_pixel_grid()
            clock.tick(60)
        bot.back_prop()
        bot.reset()
        plr.reset()
        run = True
    return

if __name__ == "__main__":
    user = False
    while(user == True and run):
        update()
        paint()
        clock.tick(60)
    if(user == False):
        update_pixel_grid()
        bot_train()
    pygame.quit() 