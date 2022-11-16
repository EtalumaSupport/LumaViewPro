#!/usr/bin/python3

import struct
import sys

def test(addr: int, data: int):
    tx_data = struct.pack('>Bi', addr, data)
    print(tx_data)

test(0x0000, 0x00000000)
test(0x0000, 0x00000001)
test(0x0000, 4294967296)
test(0x0000, -2000000000)
test(0x0000, 4294967296-2000000000)
test(0x0000, 1)
test(0x0000, -2000000000)
test(0x0000, 4294967296-2000000000)

print(
