import pygame
import random
import torch
import numpy as np
from collections import deque

import utility as util

class GameConfig:

    def __init__(
        self,
        row=10,
        col=10,
        pixel_size=20
    ):
        self.SCR_ROW = row
        self.SCR_COLUMN = col
        self.PIXEL_SIZE = pixel_size
        self.SCR_WIDTH = self.SCR_COLUMN*self.PIXEL_SIZE
        self.SCR_HEIGHT = self.SCR_ROW*self.PIXEL_SIZE
        self.SCR_WIDTH_PIXEL = int(self.SCR_WIDTH/self.PIXEL_SIZE)
        self.SCR_HEIGHT_PIXEL = int(self.SCR_HEIGHT/self.PIXEL_SIZE)

class Apple:
    scale = (1, 1)
    color = (255, 0, 0)
    
    def __init__(self, config):
        self.set_config(config)
        self.onScreen = False
        self.ava_pos = {}
        self.generate([])

    def set_config(self, config):
        self.width = self.scale[0]*config.PIXEL_SIZE
        self.height = self.scale[1]*config.PIXEL_SIZE
        self.SCR_WIDTH = config.SCR_WIDTH
        self.SCR_HEIGHT = config.SCR_HEIGHT
        self.SCR_WIDTH_PIXEL = config.SCR_HEIGHT_PIXEL
        self.SCR_HEIGHT_PIXEL = config.SCR_HEIGHT_PIXEL

    def reset(self):
        self.onScreen = False
        self.generate([])
        return

    def getX(self):
        if(self.rect == None): return
        return self.rect.x
    
    def getY(self):
        if(self.rect == None): return
        return self.rect.y

    def get_pixelX(self):
        if(self.rect == None): return
        return int(round(self.getX()/self.width))
    
    def get_pixelY(self):
        if(self.rect == None): return
        return int(round(self.getY()/self.height))

    def generate(self, occupied):
        if(self.onScreen): return
        self.onScreen = True
        self.ava_pos.clear()
        for rect in occupied:
            self.ava_pos[str(rect[0]) + " " + str(rect[1])] = True
        if(len(occupied) == self.SCR_HEIGHT_PIXEL*self.SCR_WIDTH_PIXEL): 
            self.rect = None
            return
        x = random.randint(0, self.SCR_WIDTH_PIXEL-1)
        y = random.randint(0, self.SCR_HEIGHT_PIXEL-1)
        key = str(x) + " " + str(y)
        while(key in self.ava_pos):
            x = random.randint(0, self.SCR_WIDTH_PIXEL-1)
            y = random.randint(0, self.SCR_HEIGHT_PIXEL-1)
            key = str(x) + " " + str(y)
        self.rect = pygame.Rect((x*self.width, y*self.height, self.width, self.height))

    def collide(self, x, y):
        if(self.rect == None): return
        if(self.rect.x == x and self.rect.y == y):
            self.onScreen = False
            return True
        return False

    def draw(self, screen):
        if(self.rect == None): return
        pygame.draw.rect(screen, self.color, self.rect)


class Player:
    speed = 4
    scale = (1, 1)
    color = (20, 190, 20)
    dir = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    def __init__(self, config):
        self.set_config(config)
        self.alive = True
        self.collision = True
        self.size = 1
        self.score = 0
        self.default_key = 3
        self.current_dir = self.default_key
        self.rect = []
        self.rect.append([self.current_dir, pygame.Rect((self.origin_x, self.origin_y, self.width, self.height))])
        self.time=self.SCR_HEIGHT_PIXEL*self.SCR_WIDTH_PIXEL

    def set_config(self, config):
        self.width = self.scale[0]*config.PIXEL_SIZE
        self.height = self.scale[1]*config.PIXEL_SIZE
        self.SCR_WIDTH = config.SCR_WIDTH
        self.SCR_HEIGHT = config.SCR_HEIGHT
        self.SCR_WIDTH_PIXEL = config.SCR_HEIGHT_PIXEL
        self.SCR_HEIGHT_PIXEL = config.SCR_HEIGHT_PIXEL
        self.origin_x = int(round(self.SCR_WIDTH_PIXEL/2))*config.PIXEL_SIZE
        self.origin_y = int(round(self.SCR_HEIGHT_PIXEL/2))*config.PIXEL_SIZE

    def reset(self):
        self.alive = True
        self.size = 1
        self.score = 0
        self.current_dir = self.default_key
        self.rect.clear()
        self.rect.append([self.current_dir, pygame.Rect((self.origin_x, self.origin_y, self.width, self.height))])
        self.time+=self.SCR_HEIGHT_PIXEL*self.SCR_WIDTH_PIXEL
    
    def getX(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][1].x
    
    def getY(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][1].y

    def get_pixelX(self, index):
        return int(round(self.getX(index)/self.width))
    
    def get_pixelY(self, index):
        return int(round(self.getY(index)/self.height))
    
    def get_body_pixel(self):
        return [(self.get_pixelX(i), self.get_pixelY(i)) for i in range(self.size)]
    
    def complete_movement(self):
        if(self.alive == False): return True
        if(self.rect[0][1].x%self.width or self.rect[0][1].y%self.height): return False
        return True

    def collide(self, pixelX, pixelY):
        if(self.alive == False): return
        if(self.collision == False): return False
        body = self.get_body_pixel()
        for e in body: 
            if(e[0]==self.get_pixelX(0)+pixelX and e[1]==self.get_pixelY(0)+pixelY): return True
        return False
    
    def change_dir(self, key):
        if(self.alive == False): return
        if(not self.complete_movement()): return False
        self.current_dir = key
        return True

    def move(self):
        if(self.alive == False): return
        if(self.complete_movement()):
            for i in range(self.size-1, 0, -1):
                self.rect[i][0] = self.rect[i-1][0]
            self.rect[0][0] = self.current_dir
            self.time-=1
        dx = self.dir[self.rect[0][0]][0]*self.speed
        dy = self.dir[self.rect[0][0]][1]*self.speed
        if(self.collision == True):
            if(self.rect[0][1].x+dx<0 or self.rect[0][1].x+self.width+dx>self.SCR_WIDTH or self.rect[0][1].y+dy<0 or self.rect[0][1].y+self.height+dy>self.SCR_HEIGHT):
                self.alive = False
                return
            px = self.rect[0][1].x+dx+self.width/2-((self.rect[0][1].x+dx+self.width/2)%self.width)
            py = self.rect[0][1].y+dy+self.height/2-((self.rect[0][1].y+dy+self.height/2)%self.height)
            for i in range(1, self.size, 1):
                x = self.rect[i][1].x+self.width/2-((self.rect[i][1].x+self.width/2)%self.width)
                y = self.rect[i][1].y+self.height/2-((self.rect[i][1].y+self.height/2)%self.height)
                if(px == x and py == y):
                    self.alive = False
                    return
        
        for rect in self.rect:
            dx = self.dir[rect[0]][0]*self.speed
            dy = self.dir[rect[0]][1]*self.speed
            rect[1].move_ip(dx, dy)
        return
    
    def grow(self, eaten):
        if(self.alive == False): return False
        if(not eaten): return False
        self.rect.append([self.rect[self.size-1][0], pygame.Rect((self.rect[self.size-1][1].x-self.dir[self.rect[self.size-1][0]][0]*self.width, self.rect[self.size-1][1].y-self.dir[self.rect[self.size-1][0]][1]*self.height, self.width, self.height))])
        self.size+=1
        self.time+=self.SCR_HEIGHT_PIXEL*self.SCR_WIDTH_PIXEL
        return True

    def draw(self, screen):
        for e in self.rect:
            pygame.draw.rect(screen, self.color, e[1])

class Game:
    key_W = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
    key_A = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
    key_S = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
    key_D = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
    keys = [key_W, key_A, key_S, key_D]
    
    def __init__(self, config):
        self.set_config(config)
        self.screen = pygame.display.set_mode((self.SCR_WIDTH, self.SCR_HEIGHT))
        self.plr = Player(config)
        self.apple = Apple(config)
        self.key_order = util.Queue()
        self.prev_dist = util.calculate_dist((self.plr.get_pixelX(0), self.plr.get_pixelY(0)), (self.apple.get_pixelX(), self.apple.get_pixelY()))
        self.prev_plr_size = 1
        self.display = 1
        self.clock = pygame.time.Clock()
        self.frames_stack = 4
        self.frames = deque([self.screenshot() for i in range(self.frames_stack)], maxlen=self.frames_stack)
        self.INPUT_SHAPE = self.get_frames().shape
        self.OUTPUT_SHAPE = len(self.keys)
        self.set_text()
        self.pause = False
        return
    
    def set_text(self):
        self.font = pygame.font.Font(None, 16)
        self.text_content = f"Score: {self.plr.size}"
        self.text = self.font.render(self.text_content, True, (255, 255, 255))
        self.text_rect = self.text.get_rect(left=3, top=3)
        self.pause_text = self.font.render("Press \'P\' to unpause.", True, (255, 255, 255))
        self.pause_text_rect = self.pause_text.get_rect(center=(self.SCR_WIDTH//2, self.SCR_HEIGHT//2))
    
    def set_config(self, config):
        self.SCR_WIDTH = config.SCR_WIDTH
        self.SCR_HEIGHT = config.SCR_HEIGHT
        self.SCR_WIDTH_PIXEL = config.SCR_HEIGHT_PIXEL
        self.SCR_HEIGHT_PIXEL = config.SCR_HEIGHT_PIXEL
        self.PIXEL_SIZE = config.PIXEL_SIZE

    def set_display(self, display):
        self.display = display
        return

    def reset(self):
        self.plr.reset()
        self.apple.reset()
        self.prev_plr_size = 1
        self.prev_dist = util.calculate_dist((self.plr.get_pixelX(0), self.plr.get_pixelY(0)), (self.apple.get_pixelX(), self.apple.get_pixelY()))
        self.frames.clear()
        for i in range(self.frames_stack): self.frames.append(self.screenshot())
        self.text_content = f"Score: {self.plr.size}"
        self.text = self.font.render(self.text_content, True, (255, 255, 255))
        return
    
    def get_reward(self):
        reward = 0
        plr_tuple = (self.plr.get_pixelX(0), self.plr.get_pixelY(0))
        apple_tuple = (self.apple.get_pixelX(), self.apple.get_pixelY())
        if(self.apple.rect != None): current_dist = util.calculate_dist(plr_tuple, apple_tuple)
        else: current_dist = self.prev_dist
        if(self.plr.size>self.prev_plr_size): reward+=10
        self.prev_plr_size=self.plr.size
        if(self.plr.complete_movement()): reward-=0.05
        if(current_dist>self.prev_dist): reward-=0.5
        elif(current_dist<self.prev_dist): reward+=0.5
        if(self.plr.alive == False): reward-=20
        self.prev_dist = current_dist
        return reward

    def step(self, action, fps=0):
        pygame.event.post(self.keys[action])
        self.check_event()
        self.update()
        self.draw()
        while(not self.plr.complete_movement()):
            self.update()
            self.draw()
            if(fps>0): self.clock.tick(fps)
        return self.get_frames(), self.get_reward(), not self.plr.alive, bool(self.plr.time)
    
    def get_frames(self):
        return torch.from_numpy(np.array(self.frames))
    
    def screenshot(self):
        grey_scale_grid = torch.zeros((self.SCR_HEIGHT_PIXEL, self.SCR_WIDTH_PIXEL))
        if(self.plr.alive == False): return grey_scale_grid
        snake_body = self.plr.get_body_pixel()
        apple_body = (self.apple.get_pixelX(), self.apple.get_pixelY())
        grey_scale_grid[apple_body[1], apple_body[0]] = 0.299*self.apple.color[0]+0.587*self.apple.color[1]+0.114*self.apple.color[2]
        for j, i in snake_body:
            if(j<0 or j>=self.SCR_WIDTH_PIXEL or i<0 or i>=self.SCR_HEIGHT_PIXEL): continue
            grey_scale_grid[i, j] = 0.299*self.plr.color[0]+0.587*self.plr.color[1]+0.114*self.plr.color[2]
        return grey_scale_grid

    def check_event(self):
        for e in pygame.event.get():
            if(e.type == pygame.QUIT):
                pygame.quit()
                exit(0)
            elif(e.type == pygame.KEYDOWN):
                current_key = -1
                if(self.key_order.empty()): current_key = self.plr.rect[0][0]
                else: current_key = self.key_order.rear()
                if(e.key == pygame.K_w and current_key%2 != 0 and not self.pause):
                    self.key_order.push(0)
                elif(e.key == pygame.K_a and current_key%2 != 1 and not self.pause):
                    self.key_order.push(1)
                elif(e.key == pygame.K_s and current_key%2 != 0 and not self.pause):
                    self.key_order.push(2)
                elif(e.key == pygame.K_d and current_key%2 != 1 and not self.pause):
                    self.key_order.push(3)
                elif(e.key == pygame.K_p):
                    if(self.pause == False):
                        self.pause = True
                        self.draw()
                        while(self.pause == True): self.check_event()
                    else: self.pause = False
        return
    
    def update(self):
        if(self.plr.alive == False): return
        if(not self.key_order.empty() and self.plr.change_dir(self.key_order.front())): self.key_order.pop()
        if(self.plr.grow(self.apple.collide(self.plr.getX(0), self.plr.getY(0))) == True): 
            self.text_content = f"Score: {self.plr.size}"
            self.text = self.font.render(self.text_content, True, (255, 255, 255))
        self.plr.move()
        self.apple.generate(self.plr.get_body_pixel())
        if(self.plr.complete_movement()): self.frames.append(self.screenshot())
        return
    
    def draw(self):
        if(self.display == 0): return
        self.screen.fill((0, 0, 0))
        if(self.display == 1):
            self.apple.draw(self.screen)
            self.plr.draw(self.screen)
            self.screen.blit(self.text, self.text_rect)
            if(self.pause == True): self.screen.blit(self.pause_text, self.pause_text_rect)
        pygame.display.update()
