import pygame

class Player:
    width = 20
    height = 20
    speed = 5
    color = (0, 255, 0)
    alive = True
    last_key = 3
    dir = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    rect = []
    size = 1
    grew = False

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.rect.append(pygame.Rect((0, 0, self.width, self.height)))
        self.tail = self.rect[0]

    def load(self):
        self.rect.clear()
        self.rect.append(pygame.Rect((0, 0, self.width, self.height)))
        self.tail = self.rect[0]
        self.alive = True
        self.grow = False
        self.size = 1
        self.last_key = 3
        self.speed = 5

    def changeDir(self, key):
        if(self.rect[0].x%self.width or self.rect[0].y%self.height):
            return False
        self.last_key = key
        return True

    def move(self, key):
        dx = self.dir[key][0]*self.speed
        dy = self.dir[key][1]*self.speed
        if(self.rect[0].x+dx<0 or self.rect[0].x+self.width+dx>self.SCR_WIDTH or self.rect[0].y+dy<0 or self.rect[0].y+self.height+dy>self.SCR_HEIGHT):
            self.alive = False
        else:
            tail = pygame.Rect((self.rect[self.size-1].x, self.rect[self.size-1].y, self.width, self.height))
            for i in range(self.size-1, 0, -1):
                self.rect[i] = pygame.Rect((self.rect[i-1].x, self.rect[i-1].y, self.width, self.height))
            self.rect[0].move_ip(dx, dy)
            self.last_key = key
            if(self.grew):
                if(tail.x != self.rect[self.size-1].x):
                    diff = tail.x-self.rect[self.size-1].x
                    if(diff): tail.x-=20+diff
                    else: tail.x+=20+diff
                else:
                    diff = tail.y-self.rect[self.size-1].y
                    if(diff): tail.y-=20+diff
                    else: tail.y+=20+diff
                self.rect.append(tail)
                self.size+=1
                self.grew = False
                print(str(self.rect[self.size-1].x-self.rect[self.size-2].x) + " " + str(self.rect[self.size-1].y-self.rect[self.size-2].y))
        return
    
    def grow(self, eaten):
        if(not eaten): return
        self.grew = True
        return

    def draw(self, screen):
        index = 0
        for rect in self.rect:
            index+=1
            pygame.draw.rect(screen, (0, 255-index*20, 0), rect)