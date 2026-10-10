"""The process the memory-cap guard test runs under a small cap.

Not collected by the suite (the name is not test_*.py); tests/guards/
test_memory_cap.py names it on a pytest command line, where an explicit
path is collected whatever its name.
"""

import time


def test_allocates_and_holds():
    block = bytearray(512 * 1024 * 1024)
    for i in range(0, len(block), 4096):
        block[i] = 1
    time.sleep(6)
    assert block[0] == 1
