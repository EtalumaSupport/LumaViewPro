# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run starts through a run_* member, the one public way in.

``ProtocolRunner`` forwarded the engine's ``prepare`` and ``start``, a
second public way to start a run that skipped the members' config
assembly, and one REST would have published. They are gone; this pins it.
"""

from modules.protocol_runner import ProtocolRunner


def test_the_runner_has_no_second_way_to_start_a_run():
    for name in ('prepare', 'start'):
        assert not hasattr(ProtocolRunner, name), (
            f'ProtocolRunner.{name} is back: a run starts through a run_* member'
        )
