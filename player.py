import pygame

class Player:
    width = 20
    height = 20
    speed = 5
    color = (0, 255, 0)
    alive = True
    dir = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    rect = []
    size = 1
    default_key = 3
    last_dir = default_key
    chance = 1
    collide = True

    def __init__(self, SCR_WIDTH, SCR_HEIGHT):
        self.SCR_WIDTH = SCR_WIDTH
        self.SCR_HEIGHT = SCR_HEIGHT
        self.rect.append([self.default_key, pygame.Rect((0, 0, self.width, self.height))])
        self.tail = self.rect[0]

    def getPixelX(self, index):
        return int(self.rect[index][1].x/self.width)
    
    def getPixelY(self, index):
        return int(self.rect[index][1].y/self.height)
    
    def copyPlayer(self, rect, size, last_dir, chance):
        self.rect = rect
        self.size = size
        self.last_dir = last_dir
        self.chance = chance
        return
    
    def completeMovement(self):
        if(self.rect[0][1].x%self.width or self.rect[0][1].y%self.height):
            return False
        return True

    def collideBody(self, pixelX, pixelY):
        if(self.collide == False): return False
        list = self.getBodyPixel(-1)
        for e in list:
            if(e[0]==self.getPixelX(0)+pixelX and e[1]==self.getPixelY(0)+pixelY):
                return True
        return False
    
    def reset(self):
        self.tail = self.rect[0]
        self.alive = True
        self.size = 1
        self.speed = 5
        self.rect.clear()
        self.rect.append([self.default_key, pygame.Rect((0, 0, self.width, self.height))])

    def changeDir(self, key):
        if(self.rect[0][1].x%self.width or self.rect[0][1].y%self.height):
            return False
        self.rect[0][0] = key
        return True

    def move(self):
        dx = self.dir[self.rect[0][0]][0]*self.speed
        dy = self.dir[self.rect[0][0]][1]*self.speed
        if(self.rect[0][1].x+dx<0 or self.rect[0][1].x+self.width+dx>self.SCR_WIDTH or self.rect[0][1].y+dy<0 or self.rect[0][1].y+self.height+dy>self.SCR_HEIGHT):
            if(self.collide == True):
                if(self.chance):
                    self.chance = False
                    return
                self.alive = False
            return
        if(self.collide == True):
            px = self.rect[0][1].x+dx+self.width/2-((self.rect[0][1].x+dx+self.width/2)%self.width)
            py = self.rect[0][1].y+dy+self.height/2-((self.rect[0][1].y+dy+self.height/2)%self.height)
            for i in range(1, self.size, 1):
                x = self.rect[i][1].x+self.width/2-((self.rect[i][1].x+self.width/2)%self.width)
                y = self.rect[i][1].y+self.height/2-((self.rect[i][1].y+self.height/2)%self.height)
                if(px == x and py == y):
                    if(self.chance):
                        self.chance = False
                        return
                    self.alive = False
                    return
        self.chance = True
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
        if(not eaten): return False
        self.rect.append([self.last_dir, pygame.Rect((self.rect[self.size-1][1].x-self.dir[self.last_dir][0]*self.width, self.rect[self.size-1][1].y-self.dir[self.last_dir][1]*self.height, self.width, self.height))])
        self.size+=1
        return True

    def draw(self, screen):
        for e in self.rect:
            pygame.draw.rect(screen, self.color, e[1])