# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Which installed distributions a frozen build bundles, read from its files.

The release spec ships each bundled distribution's dist-info folder, with
its licence files. Which distributions those are is read here from the
files the Analysis collected: each source file is matched to the
distribution whose installed file list (its RECORD) holds it. A name-based
match cannot do this: a distribution's top_level.txt need not name the
folders its files ship in (pywin32's names neither ``win32`` nor
``pywin32_system32``), so its files looked like LVP's own and shipped
without a licence.
"""

from __future__ import annotations

import importlib.metadata
import os
import sysconfig
from collections.abc import Iterable


def _path_key(path) -> str:
    return os.path.normcase(os.path.realpath(path))


def bundled_distributions(
    source_paths: Iterable[str],
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
    site_packages: Iterable[str] | None = None,
) -> list[str]:
    """Return the sorted names of the distributions that own the bundled files.

    A bundled file inside ``site_packages`` that no distribution claims
    refuses the build, naming the file: its licence cannot be found. A file
    outside them (LVP's own, the interpreter's) is not a distribution's.
    ``distributions`` defaults to the build environment's installed ones,
    ``site_packages`` to its purelib and platlib directories.
    """
    if distributions is None:
        distributions = importlib.metadata.distributions()
    if site_packages is None:
        paths = sysconfig.get_paths()
        site_packages = {paths['purelib'], paths['platlib']}
    roots = [_path_key(root) + os.sep for root in site_packages]

    owner_by_file: dict[str, set[str]] = {}
    for dist in distributions:
        name = dist.metadata['Name']
        for file in dist.files or ():
            owner_by_file.setdefault(_path_key(dist.locate_file(file)), set()).add(name)

    owners: set[str] = set()
    unclaimed: list[str] = []
    for source in source_paths:
        # A namespace package's entry names no file ('-'): nothing to own.
        if not source or not os.path.isfile(source):
            continue
        key = _path_key(source)
        claimed = owner_by_file.get(key)
        if claimed:
            owners |= claimed
        elif any(key.startswith(root) for root in roots):
            unclaimed.append(source)
    if unclaimed:
        raise SystemExit(
            f'FATAL: licence collection: {len(unclaimed)} bundled file(s) in '
            f'site-packages belong to no installed distribution, first '
            f'{unclaimed[0]}; their licence cannot be found.'
        )
    return sorted(owners)
