import pygame
import random

class Apple:
    width = 20
    height = 20
    color = (255, 0, 0)
    cd_count = 0
    cd_time = 180
    apple_list = []

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT

    def generate(self):
        self.cd_count = (self.cd_count+1)%self.cd_time
        if(self.cd_count): return
        x = random.randint(0, int(self.SCR_WIDTH/self.width))*self.width
        y = random.randint(0, int(self.SCR_HEIGHT/self.height))*self.height
        print(x, y)
        self.apple_list.append(pygame.Rect((x, y, self.width, self.height)))

    def draw(self, screen):
        for apple in self.apple_list:
            pygame.draw.rect(screen, self.color, apple)