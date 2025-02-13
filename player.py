import pygame

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