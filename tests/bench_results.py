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


def bench_results_dir() -> pathlib.Path:
    """``<log folder>/bench_results``, made if missing."""
    folder = get_log_dir() / 'bench_results'
    folder.mkdir(parents=True, exist_ok=True)
    return folder
