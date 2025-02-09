import pygame
import numpy as np
from player import Player
from apple import Apple
from utility import Queue

class Environment:
    pixel_size = 20
    SCR_WIDTH = 800
    SCR_HEIGHT = 600
    SCR_WIDTH_PIXEL = int(SCR_WIDTH/pixel_size)
    SCR_HEIGHT_PIXEL = int(SCR_HEIGHT/pixel_size)
    
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
    
    def getDistBAP(self):
        return np.sqrt((self.plr.getX(0)-self.apple.getX())*(self.plr.getX(0)-self.apple.getX())+(self.plr.getY(0)-self.apple.getY())*(self.plr.getY(0)-self.apple.getY()))
    
    def getNextState(self):
        list = []
        for i in range(len(self.plr.dir)):
            next_state = self.new_state()
            next_plr = Player(self.SCR_WIDTH, self.SCR_HEIGHT)
            next_plr.copyPlayer(self.plr)
            next_apple = Apple(self.SCR_WIDTH_PIXEL, self.SCR_HEIGHT)
            next_apple.copyApple(self.apple)
            next_plr.changeDir(i)
            next_plr.move()
            while(not next_plr.completeMovement()): next_plr.move()
            next_plr.grow(next_apple.collide(next_plr.getX(0), next_plr.getY(0)))
            next_body = next_plr.getBodyPixel()
            for e in next_body:
                next_state[e[1]][e[0]] = 1.0
            list.append(np.array(next_state).reshape(self.SCR_WIDTH_PIXEL*self.SCR_HEIGHT_PIXEL,))
        return list

    def getState(self):
        list = []
        for i in range(self.SCR_HEIGHT_PIXEL):
            row = []
            for j in range(self.SCR_WIDTH_PIXEL):
                row.append(self.state[i][j])
            list.append(row)
        return list

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
        self.apple.generate(self.plr.rect)
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
            for e in self.prev_body:
                self.state[e[1]][e[0]] = 0.0
            for e in self.current_body:
                self.state[e[1]][e[0]] = 1.0
            self.prev_body = self.current_body
        if(self.current_apple != self.prev_apple):
            self.state[self.current_apple[1]][self.current_apple[0]] = 1.0
            self.prev_apple = self.current_apple
        return
    
    def setup_state(self):
        self.state.clear()
        self.state = self.new_state()
        self.update_state()
        self.prev_body = self.plr.getBodyPixel()
        self.prev_apple = self.apple.getPixelTuple()
        return
    
    def new_state(self):
        list = []
        for i in range(self.SCR_HEIGHT_PIXEL):
            row = []
            for j in range(self.SCR_WIDTH_PIXEL):
                row.append(0.0)
            list.append(row)
        return list
        