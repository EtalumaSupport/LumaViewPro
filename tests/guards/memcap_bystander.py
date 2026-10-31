"""The module the memory-cap guard test runs beside the victim.

Not collected by the suite (the name is not test_*.py). Two tests, so the
loadgroup scheduler orders this unit ahead of the victim's one: the worker
completes it, then dies in the victim, and the dead worker's workload goes
back on the queue with this completed unit first.
"""


def test_one():
    assert True


def test_two():
    assert True
