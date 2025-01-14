import os
import time

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    
    
def f(name):
    print("hello bob1")
    info('function f')
    print()
    
def f2(name):
    time.sleep(5)
    print("hello bob2")
    info('function f')
    print()