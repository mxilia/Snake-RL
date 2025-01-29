from collections import deque

class Queue:
    dq = deque([])
    sz = 0

    def __init__(self):
        pass

    def push(self, x):
        self.dq.append(x)
        self.sz+=1

    def pop(self):
        if(self.sz==0): return
        self.dq.popleft()
        self.sz-=1

    def empty(self):
        if(self.sz==0): return True
        return False
    
    def front(self):
        if(self.sz==0): return -1
        x = self.dq.popleft()
        self.dq.appendleft(x)
        return x
    
    def rear(self):
        if(self.sz==0): return -1
        x = self.dq.pop()
        self.dq.append(x)
        return x