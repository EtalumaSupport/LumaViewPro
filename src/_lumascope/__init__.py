# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The API's code, one private package; what in it is API is decided by its marks.

Its import is the one point every host passes first, so a source run whose
installed packages are not the ones ``requirements.txt`` pins is refused
here, before anything third-party loads (``lib/installed_packages.py``).
"""

from _lumascope.lib.build_files import frozen
from _lumascope.lib.installed_packages import (
    PackagesNotAsPinnedError,
    packages_unlike_the_pins,
    running_python,
)

if not frozen() and (_rows := packages_unlike_the_pins()):
    raise PackagesNotAsPinnedError.not_as_pinned(_rows, running_python())
