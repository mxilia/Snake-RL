import pygame

class Player:
    player = pygame.Rect((300, 300, 50 ,50))
    color = (255, 0, 0)

    def __init__(self):
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.player)