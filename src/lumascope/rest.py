# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The headless REST server: ``python -m lumascope.rest [--simulate]``, its one command.

Serves the API over HTTP where ``rest_api`` in the settings says, through
the server's one entry (``_lumascope.rest.headless``); ``--help`` lists
what it takes. It publishes no name.
"""

import sys

from _lumascope.rest.headless import main

if __name__ == '__main__':
    sys.exit(main())
