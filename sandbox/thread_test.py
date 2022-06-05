# modified from https://realpython.com/intro-to-python-threading/

import threading
import time

t = 0

def thread_function():
    for i in range(10):
        print("Thread")
        time.sleep(0.5)

x = threading.Thread(target=thread_function)
x.start()

for i in range(20):
    print("Main")
    time.sleep(0.5)
