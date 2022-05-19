n_int = 33
n_hex = 0x21
n_bin = 0b00100001

print('int', n_int)
print('hex', n_hex)
print('bin', n_bin)

rx = b'!\x01\x00\x00\x00'

status = bin(rx[0])
# status = int.from_bytes(rx[0], byteorder='big', signed=True)
data = int.from_bytes(rx[1:5], byteorder='big', signed=True)

print(status, data)


#
# import numpy as np
#
# spi_bin = bin(list(rx)[0])
#
# spi_status = np.zeros(8)
# for i in np.arange(len(spi_bin)-1, 1, -1):
#     spi_status[i] = spi_bin[i]
#
# print(spi_bin)
# print(spi_status)
