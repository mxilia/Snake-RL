import os
import numpy as np
from collections import deque

class Queue:

    def __init__(self):
        self.dq = deque([])
        self.sz = 0

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
    
    def copy_queue(self, queue):
        while(not queue.empty()):
            self.push(queue.front())
            queue.pop()
        return
    
def calculate_dist(a, b):
    return np.sqrt(np.square(a[0]-b[0])+np.square(a[1]-b[1]))

def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Create {directory_name} successfully.")
    except FileExistsError:
        print(f"{directory_name} existed.\nAre you sure you want to use the same name? (If you're testing then just pick y)\n y/n")
        while(True):
            c = input()
            if(c=='n'): exit(0)
            elif(c=='y'): break
        return
    except PermissionError:
        print(f"Creating {directory_name} denied.")
        return
    except Exception as e:
        print(f"Creating {directory_name} error.")
        return
