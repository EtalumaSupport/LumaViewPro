# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where a bench run's results go: ``bench_results`` in the one log folder.

A bench run is anything run against the connected scope: a capability probe
under ``--hardware``, or a pytest run with a ``--run-*-hardware`` flag. Its
verdicts, records, driver log and results file all land here, beside the
app's own logs, so the support zip carries them; the simulator suite's
writes stay in temporary folders.
"""

import pathlib

from modules.path_utils import get_log_dir

REPO = pathlib.Path(__file__).resolve().parent.parent

# What a pytest run, a probe or the real logger needs from a checkout.
_CHECKOUT_ITEMS = (
    'lvp_logger.py',
    'version.txt',
    'pyproject.toml',
    'modules',
    'drivers',
    'lib',
    'tests',
    'data',
)


def bench_results_dir() -> pathlib.Path:
    """``<log folder>/bench_results``, made if missing."""
    folder = get_log_dir() / 'bench_results'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def linked_checkout(parent: pathlib.Path) -> pathlib.Path:
    """A folder of links into this checkout, under ``parent``, for a child run.

    The log folder follows the folder a run is launched from, links not
    followed, so a child run started here (a bench-flagged pytest, a probe
    under ``--hardware``) writes its logs and results into this folder and
    never into the checkout's own log folder. Start the child with this
    folder as its working directory: a process started inside a linked
    directory is handed the clone's path instead.
    """
    folder = parent / 'checkout'
    folder.mkdir(parents=True)
    for item in _CHECKOUT_ITEMS:
        (folder / item).symlink_to(REPO / item)
    return folder
