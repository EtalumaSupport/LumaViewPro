"""psutil is the installed library inside a test run.

A MagicMock standing in for psutil in sys.modules reaches every reader of
it, including pytest-xdist, whose auto worker count is psutil.cpu_count();
a mock there is truthy, builds zero workers, and `pytest -n auto` exits 4
having run nothing.
"""

import psutil


def test_psutil_cpu_count_is_an_int_inside_a_run():
    assert type(psutil.cpu_count()) is int
