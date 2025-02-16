import pygame
import numpy as np
from utility import Queue
import random

class Apple:
    width = 20
    height = 20
    color = (255, 0, 0)
    
    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.boundPixelX = int(self.SCR_WIDTH/self.width)
        self.boundPixelY = int(self.SCR_HEIGHT/self.height)
        self.onScreen = False
        self.rect = pygame.Rect((500, 500, self.width, self.height))
        self.ava_pos = {}

    def getX(self):
        return self.rect.x
    
    def getY(self):
        return self.rect.y

    def getPixelX(self):
        return int(self.rect.x/self.width)
    
    def getPixelY(self):
        return int(self.rect.y/self.height)
    
    def getPixelTuple(self):
        return (self.getPixelX(), self.getPixelY())
    
    def copyApple(self, apple):
        if(apple.onScreen == False): return
        self.rect = pygame.Rect((apple.rect.x, apple.rect.y, apple.rect.width, apple.rect.height))
        return

    def generate(self, occupied):
        if(self.onScreen): return
        self.onScreen = True
        self.ava_pos.clear()
        for rect in occupied:
            self.ava_pos[str(rect[0]) + " " + str(rect[1])] = True
        x = random.randint(0, self.boundPixelX-1)
        y = random.randint(0, self.boundPixelY-1)
        key = str(x) + " " + str(y)
        while(key in self.ava_pos):
            x = random.randint(0, self.boundPixelX-1)
            y = random.randint(0, self.boundPixelY-1)
            key = str(x) + " " + str(y)
        self.rect = pygame.Rect((x*self.width, y*self.height, self.width, self.height))

    def collide(self, x, y):
        if(self.rect.x == x and self.rect.y == y):
            self.onScreen = False
            return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)


class Player:
    width = 20
    height = 20
    speed = 4
    color = (0, 255, 0)
    dir = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.alive = True
        self.rect = []
        self.size = 1
        self.default_key = 3
        self.collide = True
        self.move_cnt = 0
        self.rect.append([self.default_key, pygame.Rect((SCR_WIDTH/2-self.width, SCR_HEIGHT/2-self.height, self.width, self.height))])

    def reset(self):
        self.alive = True
        self.size = 1
        self.move_cnt = 0
        self.rect.clear()
        self.rect.append([self.default_key, pygame.Rect((self.SCR_WIDTH/2-self.width, self.SCR_HEIGHT/2-self.height, self.width, self.height))])

    def dead(self):
        self.alive = False
        self.size = 0
        self.rect.clear()
        return
    
    def getMoveCnt(self):
        return self.move_cnt
    
    def getX(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][1].x
    
    def getY(self, index):
        if(index>=self.size or index<0): return None
        return self.rect[index][1].y

    def getPixelX(self, index):
        if(index>=self.size or index<0): return None
        return int(self.rect[index][1].x/self.width)
    
    def getPixelY(self, index):
        if(index>=self.size or index<0): return None
        return int(self.rect[index][1].y/self.height)
    
    def getSize(self):
        return self.size
    
    def copyPlayer(self, plr):
        self.rect.clear()
        for i in range(len(plr.rect)):
            self.rect.append([plr.rect[i][0], plr.rect[i][1].copy()])
        self.size = plr.size
        return
    
    def getBodyPixel(self):
        list = []
        for i in range(self.size):
            list.append((self.getPixelX(i), self.getPixelY(i)))
        return list
    
    def completeMovement(self):
        if(self.alive == False): return True
        if(self.rect[0][1].x%self.width or self.rect[0][1].y%self.height):
            return False
        return True

    def collideBody(self, pixelX, pixelY):
        if(self.alive == False): return
        if(self.collide == False): return False
        list = self.getBodyPixel()
        for e in list:
            if(e[0]==self.getPixelX(0)+pixelX and e[1]==self.getPixelY(0)+pixelY):
                return True
        return False
    
    def changeDir(self, key):
        if(self.alive == False): return
        if(self.rect[0][1].x%self.width or self.rect[0][1].y%self.height):
            return False
        self.rect[0][0] = key
        return True

    def move(self):
        if(self.alive == False): return
        dx = self.dir[self.rect[0][0]][0]*self.speed
        dy = self.dir[self.rect[0][0]][1]*self.speed
        if(self.collide == True):
            if(self.rect[0][1].x+dx<0 or self.rect[0][1].x+self.width+dx>self.SCR_WIDTH or self.rect[0][1].y+dy<0 or self.rect[0][1].y+self.height+dy>self.SCR_HEIGHT):
                self.dead()
                return
            px = self.rect[0][1].x+dx+self.width/2-((self.rect[0][1].x+dx+self.width/2)%self.width)
            py = self.rect[0][1].y+dy+self.height/2-((self.rect[0][1].y+dy+self.height/2)%self.height)
            for i in range(1, self.size, 1):
                x = self.rect[i][1].x+self.width/2-((self.rect[i][1].x+self.width/2)%self.width)
                y = self.rect[i][1].y+self.height/2-((self.rect[i][1].y+self.height/2)%self.height)
                if(px == x and py == y):
                    self.dead()
                    return
        if(self.completeMovement()): self.move_cnt+=1
        for rect in self.rect:
            dx = self.dir[rect[0]][0]*self.speed
            dy = self.dir[rect[0]][1]*self.speed
            rect[1].move_ip(dx, dy)
        self.last_dir = self.rect[self.size-1][0]
        if(self.rect[0][1].x%self.width == 0 and self.rect[0][1].y%self.height == 0):
            for i in range(self.size-1, 0, -1):
                self.rect[i][0] = self.rect[i-1][0]
        return
    
    def grow(self, eaten):
        if(self.alive == False): return False
        if(not eaten): return False
        self.rect.append([self.last_dir, pygame.Rect((self.rect[self.size-1][1].x-self.dir[self.last_dir][0]*self.width, self.rect[self.size-1][1].y-self.dir[self.last_dir][1]*self.height, self.width, self.height))])
        self.size+=1
        return True

    def draw(self, screen):
        for e in self.rect:
            pygame.draw.rect(screen, self.color, e[1])

class Snake_Game:
    pixel_size = 20
    SCR_WIDTH = 800
    SCR_HEIGHT = 600
    SCR_WIDTH_PIXEL = int(SCR_WIDTH/pixel_size)
    SCR_HEIGHT_PIXEL = int(SCR_HEIGHT/pixel_size)

    key_W = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
    key_A = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
    key_S = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
    key_D = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)
    keys = [key_W, key_A, key_S, key_D]
    
    def __init__(self):
        self.screen = pygame.display.set_mode((self.SCR_WIDTH, self.SCR_HEIGHT))
        self.key_order = Queue()
        self.plr = Player(self.SCR_WIDTH, self.SCR_HEIGHT)
        self.apple = Apple(self.SCR_WIDTH, self.SCR_HEIGHT)
        self.state = []
        self.prev_body = []
        self.prev_apple = ()
        self.setup_state()
        return

    def reset(self):
        self.plr.reset()
        self.setup_state()
        return
    
    def getReward(self, mul=10):
        reward = self.plr.getMoveCnt()*0.5+self.plr.getSize()*mul
        if(self.plr.alive == False): reward-=100
        return reward
    
    def getDist(self):
        return np.sqrt((self.plr.getX(0)-self.apple.getX())*(self.plr.getX(0)-self.apple.getX())+(self.plr.getY(0)-self.apple.getY())*(self.plr.getY(0)-self.apple.getY()))
    
    def emulate(self, action):
        next_state = self.new_state()
        next_plr = Player(self.SCR_WIDTH, self.SCR_HEIGHT)
        next_plr.copyPlayer(self.plr)
        next_apple = Apple(self.SCR_WIDTH_PIXEL, self.SCR_HEIGHT)
        next_apple.copyApple(self.apple)
        next_plr.changeDir(action)
        next_plr.move()
        while(not next_plr.completeMovement()): next_plr.move()
        next_plr.grow(next_apple.collide(next_plr.getX(0), next_plr.getY(0)))
        next_body = next_plr.getBodyPixel()
        for e in next_body:
            next_state[e[1]][e[0]] = 1.0
        return next_state

    def getState(self):
        list = []
        for i in range(self.SCR_HEIGHT_PIXEL):
            row = []
            for j in range(self.SCR_WIDTH_PIXEL):
                row.append(self.state[i][j])
            list.append(row)
        return list
    
    def postAction(self, action):
        pygame.event.post(self.keys[action])
        return

    def checkEvent(self, e):
        if(e.type != pygame.KEYDOWN): return
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
        if(not self.key_order.empty() and self.plr.changeDir(self.key_order.front())):
            self.key_order.pop()
        self.plr.grow(self.apple.collide(self.plr.getX(0), self.plr.getY(0)))
        self.plr.move()
        self.apple.generate(self.plr.getBodyPixel())
        self.update_state()
        return
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        self.plr.draw(self.screen)
        self.apple.draw(self.screen)
    
    def update_state(self):
        self.current_body = self.plr.getBodyPixel()
        self.current_apple = self.apple.getPixelTuple()
        if(self.current_body != self.prev_body):
            if(self.prev_apple != None):
                for e in self.prev_body:
                    self.state[e[1]][e[0]] = -1.0
            for e in self.current_body:
                self.state[e[1]][e[0]] = 1.0
            self.prev_body = self.current_body
        if(self.current_apple != self.prev_apple):
            if(self.prev_apple != None): self.state[self.prev_apple[1]][self.prev_apple[0]] = -1.0
            self.state[self.current_apple[1]][self.current_apple[0]] = 1.0
            self.prev_apple = self.current_apple
        return
    
    def setup_state(self):
        self.state.clear()
        self.prev_body = None
        self.prev_apple = None
        self.state = self.new_state()
        self.update_state()
        return
    
    def new_state(self):
        list = []
        for i in range(self.SCR_HEIGHT_PIXEL):
            row = []
            for j in range(self.SCR_WIDTH_PIXEL):
                row.append(0.0)
            list.append(row)
        return list
        