# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 driver step's executable, which the Windows installer runs elevated."""

import sys

from modules.fx2_driver_step import main

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
