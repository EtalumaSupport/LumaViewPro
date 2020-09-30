import pyfirmata
import time

board = pyfirmata.Arduino('COM5')

r = board.digital[11]
g = board.digital[12]
b = board.digital[13]

for n in range(10):
    r.write(1)
    time.sleep(0.5)
    r.write(0)

    g.write(1)
    time.sleep(0.5)
    g.write(0)

    b.write(1)
    time.sleep(0.5)
    b.write(0)

time.sleep(10)
