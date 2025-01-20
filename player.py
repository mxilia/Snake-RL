import pygame

class Player:
    width = 20
    height = 20
    speed = 5
    color = (0, 255, 0)
    alive = True
    last_key = 3
    dir = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.player = pygame.Rect((0, 0, self.width, self.height))

    def move(self, key, pressed):
        dx = self.dir[key][0]*self.speed
        dy = self.dir[key][1]*self.speed
        if(pressed):
            if(key%2 == 0 and self.player.x%self.width and self.player.y%self.height):
                return False
            elif(key%2 == 1 and self.player.x%self.width and self.player.y%self.height):
                return False
        if(self.player.x+dx<0 or self.player.x+self.width+dx>self.SCR_WIDTH or self.player.y+dy<0 or self.player.y+self.height+dy>self.SCR_HEIGHT):
            self.alive = False
        else: 
            self.player.move_ip(dx, dy)
            self.last_key = key
        return True


    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.player)