import pygame
import random
import numpy as np
from collections import deque

import utility as util


class Game_Config:
    SCR_ROW = 10
    SCR_COLUMN = 10
    PIXEL_SIZE = 20
    SCR_WIDTH = SCR_COLUMN*PIXEL_SIZE
    SCR_HEIGHT = SCR_ROW*PIXEL_SIZE
    SCR_WIDTH_PIXEL = int(SCR_WIDTH/PIXEL_SIZE)
    SCR_HEIGHT_PIXEL = int(SCR_HEIGHT/PIXEL_SIZE)

    def __init__(self):
        pass

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
        return self.rect.x
    
    def getY(self):
        return self.rect.y

    def get_pixelX(self):
        return int(round(self.rect.x/self.width))
    
    def get_pixelY(self):
        return int(round(self.rect.y/self.height))
    
    def copy(self, target):
        if(target.onScreen == False): return
        self.rect = pygame.Rect((target.rect.x, target.rect.y, self.width, self.height))
        return

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
        self.rect = []
        self.rect.append([self.default_key, pygame.Rect((self.origin_x, self.origin_y, self.width, self.height))])

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
        self.rect.clear()
        self.rect.append([self.default_key, pygame.Rect((self.origin_x, self.origin_y, self.width, self.height))])
    
    def getX(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][1].x
    
    def getY(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][1].y

    def get_pixelX(self, index):
        if(index>=self.size or index<0): return None
        return int(round(self.rect[index][1].x/self.width))
    
    def get_pixelY(self, index):
        if(index>=self.size or index<0): return None
        return int(round(self.rect[index][1].y/self.height))
    
    def get_body_pixel(self):
        body = []
        for i in range(self.size):
            body.append((self.get_pixelX(i), self.get_pixelY(i)))
        return body
    
    def get_dir(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][0]
    
    def copy(self, target):
        self.rect.clear()
        for i in range(target.size):
            self.rect.append([target.rect[i][0], target.rect[i][1].copy()])
        self.size = target.size
        return
    
    def complete_movement(self):
        if(self.alive == False): return True
        if(self.rect[0][1].x%self.width or self.rect[0][1].y%self.height): return False
        return True

    def collide(self, pixelX, pixelY):
        if(self.alive == False): return
        if(self.collide == False): return False
        body = self.get_body_pixel()
        for e in body:
            if(e[0]==self.get_pixelX(0)+pixelX and e[1]==self.get_pixelY(0)+pixelY): return True
        return False
    
    def change_dir(self, key):
        if(self.alive == False): return
        if(not self.complete_movement()): return False
        self.rect[0][0] = key
        return True

    def move(self):
        if(self.alive == False): return
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
        if(self.complete_movement()):
            for i in range(self.size-1, 0, -1):
                self.rect[i][0] = self.rect[i-1][0]
        return
    
    def grow(self, eaten):
        if(self.alive == False): return False
        if(not eaten): return False
        self.rect.append([self.rect[self.size-1][0], pygame.Rect((self.rect[self.size-1][1].x-self.dir[self.rect[self.size-1][0]][0]*self.width, self.rect[self.size-1][1].y-self.dir[self.rect[self.size-1][0]][1]*self.height, self.width, self.height))])
        self.size+=1
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
    
    def __init__(self):
        self.config = Game_Config()
        self.set_config(self.config)
        self.screen = pygame.display.set_mode((self.SCR_WIDTH, self.SCR_HEIGHT))
        self.plr = Player(self.config)
        self.apple = Apple(self.config)
        self.key_order = util.Queue()
        self.prev_dist = 1000000
        self.prev_plr_size = 1
        self.display = 1
        self.clock = pygame.time.Clock()
        self.frames_stack = 4
        self.frames = deque([self.screenshot() for i in range(self.frames_stack)], maxlen=self.frames_stack)
        self.INPUT_SHAPE = self.get_frames().shape
        self.OUTPUT_SHAPE = len(self.keys)
        return
    
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
        self.prev_dist = 1000000
        self.frames.clear()
        for i in range(self.frames_stack): self.frames.append(self.screenshot())
        return
    
    def get_reward(self):
        reward = 0
        plr_tuple = (self.plr.get_pixelX(0), self.plr.get_pixelY(0))
        apple_tuple = (self.apple.get_pixelX(), self.apple.get_pixelY())
        current_dist = util.calculate_dist(plr_tuple, apple_tuple)
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
        return self.get_frames(), self.get_reward(), not self.plr.alive
    
    def get_frames(self):
        return np.array(self.frames)
    
    def screenshot(self):
        surface = pygame.display.get_surface()
        rgb_arr = pygame.surfarray.array3d(surface)
        grey_scale_arr = np.array(0.2989*rgb_arr[:, :, 0]+0.5870*rgb_arr[:, :, 1]+0.1140*rgb_arr[:, :, 2])
        return grey_scale_arr

    def check_event(self):
        for e in pygame.event.get():
            if(e.type == pygame.QUIT):
                pygame.quit()
                exit(0)
            elif(e.type == pygame.KEYDOWN):
                current_key = -1
                if(self.key_order.empty()): current_key = self.plr.rect[0][0]
                else: current_key = self.key_order.rear()
                if(e.key == pygame.K_w and current_key%2 != 0):
                    self.key_order.push(0)
                if(e.key == pygame.K_a and current_key%2 != 1):
                    self.key_order.push(1)
                if(e.key == pygame.K_s and current_key%2 != 0):
                    self.key_order.push(2)
                if(e.key == pygame.K_d and current_key%2 != 1):
                    self.key_order.push(3)
        return
    
    def update(self):
        if(self.plr.alive == False): return
        if(not self.key_order.empty() and self.plr.change_dir(self.key_order.front())): self.key_order.pop()
        self.plr.grow(self.apple.collide(self.plr.getX(0), self.plr.getY(0)))
        self.plr.move()
        self.apple.generate(self.plr.get_body_pixel())
        self.frames.append(self.screenshot())
        return
    
    def draw(self):
        if(self.display == 0): return
        self.screen.fill((0, 0, 0))
        if(self.display == 1):
            self.apple.draw(self.screen)
            self.plr.draw(self.screen)
        elif(self.display == 2):
            state = self.get_state()
            for i in range(1, self.SCR_HEIGHT_PIXEL+1):
                for j in range(1, self.SCR_WIDTH_PIXEL+1):
                    Color = None
                    c_pix = int(state[i][j])
                    if(c_pix == -1): Color = (255, 255, 255)
                    elif(c_pix == -2): Color = (210, 200, 10)
                    elif(c_pix == 1): Color = (25, 200, 10)
                    elif(c_pix == 2): Color = (250, 20, 10)
                    else: continue
                    pygame.draw.rect(self.screen, Color, 
                    pygame.Rect(((j-1)*self.PIXEL_SIZE, (i-1)*self.PIXEL_SIZE, self.PIXEL_SIZE, self.PIXEL_SIZE)))
        pygame.display.update()

        '''
        # This was used before I implemented convo nn. Not efficient.
        def get_state(self, plr=None, apple=None):
            if(plr == None): plr=self.plr
            if(apple == None): apple=self.apple
            current_body = plr.get_body_pixel()
            current_apple = (apple.get_pixelX(), apple.get_pixelY())
            state = [[1.0 if j == 0 or i == 0 or j == self.SCR_WIDTH_PIXEL+1 or i == self.SCR_HEIGHT_PIXEL+1 else 0.0 for j in range(self.SCR_WIDTH_PIXEL+2)] for i in range(self.SCR_HEIGHT_PIXEL+2)]
            if(self.plr.alive == False): return state
            offset = 1
            state[current_apple[1]+offset][current_apple[0]+offset] = 2.0
            head = current_body[0]
            tail = current_body[plr.size-1]
            if(tail[0]>=0 and tail[0]<self.SCR_WIDTH_PIXEL and tail[1]>=0 or tail[1]<self.SCR_HEIGHT_PIXEL): state[tail[1]+offset][tail[0]+offset] = -2.0
            if(head[0]>=0 and head[0]<self.SCR_WIDTH_PIXEL and head[1]>=0 or head[1]<self.SCR_HEIGHT_PIXEL): state[head[1]+offset][head[0]+offset] = 1.0
            for i in range(1, len(current_body)-1):
                e = current_body[i]
                if(e[0]<0 or e[0]>=self.SCR_WIDTH_PIXEL  or e[1]<0 or e[1]>=self.SCR_HEIGHT_PIXEL): continue
                state[e[1]+offset][e[0]+offset] = -1.0
            return state
        '''