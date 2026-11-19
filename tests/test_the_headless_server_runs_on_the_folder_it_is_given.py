# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``python -m lumascope.rest --source-path FOLDER`` runs on that folder's ``data/``, as ``ScopeSession.create`` does.

Before, the server always ran on its own installation's files, so a
simulated server, or a client's example run against one, wrote into the
user's settings and capture folder. A folder that holds no installation is
refused naming the file it lacks, and the server exits 1.
"""

from __future__ import annotations

import json

from tests.settings_fixtures import complete_settings, installation_of_its_own


def test_the_server_reads_the_settings_of_the_folder_it_is_given(capsys, a_host_in_this_process):
    from _lumascope.rest.headless import main

    root = installation_of_its_own()
    current = complete_settings()
    # A port no server can take, written only in this folder's settings.
    current.update(lvp_lock_port=0, rest_api={**current['rest_api'], 'port': 70000})
    (root / 'data' / 'current.json').write_text(json.dumps(current))

    exited = main(['--simulate', '--source-path', str(root)])

    said = capsys.readouterr()
    assert exited == 1
    assert 'rest_api.port' in said.err
    assert 'REST server:' not in said.out


def test_a_folder_with_no_installation_is_refused_and_the_server_exits_1(
    tmp_path, capsys, a_host_in_this_process
):
    from _lumascope.rest.headless import main

    exited = main(['--simulate', '--source-path', str(tmp_path)])

    said = capsys.readouterr()
    assert exited == 1
    assert 'settings.json' in said.err
    assert 'REST server:' not in said.out
