import pygame

class Player:
    width = 20
    height = 20
    speed = 4
    color = (0, 255, 0)

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.player = pygame.Rect((SCR_WIDTH/2-self.width, SCR_HEIGHT/2-self.height, self.width, self.height))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.player)

    def move(self, dx, dy):
        if self.player.x+dx<0 or self.player.x+self.width+dx>self.SCR_WIDTH:
            return
        if self.player.y+dy<0 or self.player.y+self.height+dy>self.SCR_HEIGHT:
            return
        self.player.move_ip(dx, dy)