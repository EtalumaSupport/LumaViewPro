# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``python -m lumascope.rest`` reports a close that failed once, through the one reporter, with its traceback.

The close's raise ends its flight in the launcher, which before logged only
its repr, so the step that failed was in the log without the traceback that
says where.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import uvicorn

from _lumascope.modules.exceptions import ScopeDisconnectError
from tests.settings_fixtures import complete_settings, installation_of_its_own


def test_a_close_that_failed_is_reported_once_with_its_traceback(
    monkeypatch, caplog, a_host_in_this_process
):
    import _lumascope.rest.server
    from _lumascope.rest.app import build_app
    from _lumascope.rest.headless import main
    from _lumascope.rest.server import Server

    root = installation_of_its_own()
    (root / 'data' / 'current.json').write_text(json.dumps(complete_settings(lvp_lock_port=0)))
    served = []

    def serve_and_close(session, *, shut_down):
        # The real server, on a port of its own, closed by the host's own
        # close once it serves; one shutdown step of the real scope fails.
        # a stand-in by design: the simulator has no fault that fails a
        # board's own disconnect.
        session.scope._motion_driver.disconnect = MagicMock(
            side_effect=RuntimeError('CLOSE_MARKER')
        )
        app = build_app(session)
        server = Server(
            uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'),
            session,
            app.state.closing,
            shut_down=shut_down,
        )

        def close_once_serving():
            while not server.serving:
                threading.Event().wait(0.01)
            try:
                server.close()
            except ScopeDisconnectError:
                pass  # the launcher's to report

        threading.Thread(target=close_once_serving, daemon=True).start()
        served.append(session)
        return server

    monkeypatch.setattr(_lumascope.rest.server, 'build_server', serve_and_close)

    try:
        exited = main(['--simulate', '--source-path', str(root)])
    finally:
        for session in served:
            session.shutdown()

    assert exited == 1
    reported = [
        r for r in caplog.records if r.exc_info and isinstance(r.exc_info[1], ScopeDisconnectError)
    ]
    assert len(reported) == 1
    assert caplog.text.count('RuntimeError: CLOSE_MARKER') == 1
