import serial
import time

ser = serial.Serial('COM5', 9600)  # open serial port
for i in range (20):
    ser.write(b'r')     # write a string
    time.sleep(0.1)
    ser.write(b'o')     # write a string

    ser.write(b'g')     # write a string
    time.sleep(0.1)
    ser.write(b'o')     # write a string

    ser.write(b'b')     # write a string
    time.sleep(0.1)
    ser.write(b'o')     # write a string
    print(i)

ser.close()
