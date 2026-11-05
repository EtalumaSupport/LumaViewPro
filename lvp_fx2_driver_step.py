# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 driver step's executable, which the Windows installer runs elevated."""

import os
import sys

# This checkout's own API, whatever lumascope.pth the interpreter holds; the
# link is not followed, so a link farm stays its own launch folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from _lumascope.modules.fx2_driver_step import main

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
