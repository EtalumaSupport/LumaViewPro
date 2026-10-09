# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The headless REST server: ``python -m rest [--simulate]``.

Brings the scope up as the GUI does -- the single-instance lock, the
prepared settings, ``ScopeSession.create``, the startup motion, the
plugins, the metrics -- and serves the declared API on
``127.0.0.1:<rest_api.port>``. It has no credentials, so it binds the
loopback interface only and takes no host: anyone who can reach the port
drives the scope. SIGINT or SIGTERM closes it with its clients connected
(``rest.server``); it exits 0, or 1 when the session did not close cleanly.
"""

import argparse
import sys

HOST = '127.0.0.1'
# Seconds the close waits for the calls' threads once the session has shut
# down; a call still running then is logged by name and the process exits.
JOIN_S = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m rest',
        description='Serve the Lumascope API over HTTP on 127.0.0.1.',
    )
    parser.add_argument(
        '--simulate', action='store_true', help='drive the simulated scope, not hardware'
    )
    args = parser.parse_args(argv)

    from lvp_logger import install_crash_hooks, logger

    install_crash_hooks()

    from modules import lvp_lock
    from modules.path_utils import get_source_root

    source_path = get_source_root()
    instance_lock = lvp_lock.take_instance_lock(source_path)
    if instance_lock is None:
        message = 'Another LumaViewPro or REST server is already running on this computer.'
        logger.error(f'[REST     ] {message}')
        print(message, file=sys.stderr)
        return 1

    # Imported after the lock: a second launch exits before it loads the
    # scope's modules and the server's dependencies.
    import uvicorn

    from modules.scope_session import ScopeSession
    from rest.app import build_app
    from rest.server import Server

    settings = ScopeSession.load_user_settings(str(source_path))
    port = settings['rest_api']['port']

    session = ScopeSession.create(
        settings=settings,
        source_path=str(source_path),
        simulate=args.simulate,
        warn_pre_release=False,
    )
    try:
        app = build_app(session)
        server = Server(uvicorn.Config(app, host=HOST, port=port), session, app.state.closing)
        # A failed home is reported and the motion ends; the server still
        # serves, so a client can home again.
        session.start_application_session()
        session.load_plugins()
        session.start_metrics()
        print(
            f'LumaViewPro REST server: http://{HOST}:{port}/api/v1 (reference: /docs)', flush=True
        )
        server.run()
        if not server.close_begun:
            # It stopped without its close: it did not start serving, or was stopped before it did.
            session.shutdown()
            return 1
        running = app.state.closing.join_jobs(JOIN_S)
        if running:
            logger.error(f'[REST     ] Exiting with calls still running: {running}')
        return 1 if server.close_failed is not None else 0
    except BaseException:
        session.shutdown()
        raise
    finally:
        instance_lock.close()


if __name__ == '__main__':
    sys.exit(main())
