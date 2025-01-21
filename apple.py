import pygame
import random

class Apple:
    width = 20
    height = 20
    color = (255, 0, 0)
    cd_count = -1
    cd_time = 180
    apple_list = []

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.boundPixelX = int(self.SCR_WIDTH/self.width)
        self.boundPixelY = int(self.SCR_HEIGHT/self.height)

    def generate(self):
        self.cd_count = (self.cd_count+1)%self.cd_time
        if(self.cd_count): return
        x = random.randint(0, self.boundPixelX)*self.width
        y = random.randint(0, self.boundPixelY)*self.height
        self.apple_list.append(pygame.Rect((x, y, self.width, self.height)))

    def collide(self, x, y):
        for apple in self.apple_list:
            if(apple.x == x and apple.y == y):
                self.apple_list.remove(apple)
                return True
        return False

    def draw(self, screen):
        for apple in self.apple_list:
            pygame.draw.rect(screen, self.color, apple)