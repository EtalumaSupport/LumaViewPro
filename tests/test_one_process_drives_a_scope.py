# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One process drives an installation's scope: the GUI and the REST server share one lock.

``modules.lvp_lock.take_instance_lock`` is the one way a host takes it;
``python -m rest`` (``rest.__main__``) refuses to start while it is held,
and serves on the loopback interface only, taking no host.
"""

import json
import socket

import pytest

from modules.lvp_lock import take_instance_lock
from modules.path_utils import get_source_root

# The launcher's tests take the installation's own lock, so they run on
# one worker (`--dist loadgroup`, in the pytest addopts).
pytestmark = pytest.mark.xdist_group('installation_lock')


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def test_a_second_taker_is_refused_while_the_lock_is_held(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'settings.json').write_text(json.dumps({'lvp_lock_port': _free_port()}))

    first = take_instance_lock(tmp_path)
    assert first is not None
    try:
        assert take_instance_lock(tmp_path) is None
    finally:
        first.close()

    again = take_instance_lock(tmp_path)
    assert again is not None
    again.close()


def test_the_server_does_not_start_beside_a_holder(capsys):
    from rest.__main__ import main

    held = take_instance_lock(get_source_root())
    try:
        assert main(['--simulate']) == 1
    finally:
        if held is not None:
            held.close()
    assert 'already running' in capsys.readouterr().err


def test_the_server_takes_no_host():
    from rest.__main__ import main

    with pytest.raises(SystemExit) as refused:
        main(['--host', '0.0.0.0'])
    assert refused.value.code == 2
