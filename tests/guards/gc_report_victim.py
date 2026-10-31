"""The tests the full-collection report test runs in a subprocess.

Not collected by the suite (the name is not test_*.py); tests/guards/
test_a_failed_tests_report_lists_its_full_collections.py names it on a
pytest command line.
"""

import gc


def test_a_red_after_one_full_collection():
    gc.collect()
    raise AssertionError('red on purpose')


def test_a_green_after_one_full_collection():
    gc.collect()
