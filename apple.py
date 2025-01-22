import pygame
import random

class Apple:
    width = 20
    height = 20
    color = (255, 0, 0)
    onScreen = False
    rect = pygame.Rect((1000, 1000, width, height))

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.boundPixelX = int(self.SCR_WIDTH/self.width)
        self.boundPixelY = int(self.SCR_HEIGHT/self.height)

    def generate(self):
        if(self.onScreen): return
        self.onScreen = True
        x = random.randint(0, self.boundPixelX-1)*self.width
        y = random.randint(0, self.boundPixelY-1)*self.height
        self.rect = pygame.Rect((x, y, self.width, self.height))

    def collide(self, x, y):
        if(self.rect.x == x and self.rect.y == y):
            self.onScreen = False
            return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)