# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The headless REST server: ``python -m _lumascope.rest [--simulate]``.

Brings the scope up as the GUI does -- the single-instance lock, the
prepared settings, ``ScopeSession.create``, the startup motion, the
plugins, the metrics -- and serves the declared API through
``_lumascope.rest.server.build_server`` on ``rest_api.address`` and ``rest_api.port``
(shipped ``127.0.0.1:8000``, this computer only). It has no credentials:
anyone who can reach the address drives the scope, unless
``rest_api.allowed_clients`` names the clients served. A ``rest_api``
setting that cannot be used, or an address and port that cannot be taken,
is printed and it exits 1. SIGINT or SIGTERM
closes it with its clients connected (``_lumascope.rest.server``); it exits 0, or 1
when the session did not close cleanly.
"""

import argparse
import sys


def main(argv: list[str] | None = None, *, prog: str = 'python -m _lumascope.rest') -> int:
    """Run the server until it is closed; *prog* is the command its usage names."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Serve the Lumascope API over HTTP, where rest_api in the settings says.',
    )
    parser.add_argument(
        '--simulate', action='store_true', help='drive the simulated scope, not hardware'
    )
    args = parser.parse_args(argv)

    from _lumascope.lvp_logger import install_crash_hooks, log_dir, logger

    install_crash_hooks()

    # An installed server takes the installer's logs from TEMP, as the GUI
    # does; any other run captures nothing.
    from _lumascope.modules.app_environment import capture_installer_logs

    capture_installer_logs(log_dir)

    from _lumascope.modules import lvp_lock
    from _lumascope.modules.path_utils import get_source_root

    source_path = get_source_root()
    instance_lock = lvp_lock.take_instance_lock(source_path)
    if instance_lock is None:
        message = 'Another LumaViewPro or REST server is already running on this computer.'
        logger.error(f'[REST     ] {message}')
        print(message, file=sys.stderr)
        return 1

    # Imported after the lock: a second launch exits before it loads the
    # server's dependencies, and, run as ``python -m _lumascope.rest``, the
    # scope's modules too (``python -m lumascope.rest`` has loaded the API
    # before this runs).
    from _lumascope.modules.exceptions import RestServerNotStartedError
    from _lumascope.modules.scope_session import ScopeSession
    from _lumascope.rest.server import build_server

    settings = ScopeSession.load_user_settings(str(source_path))

    session = ScopeSession.create(
        settings=settings,
        source_path=str(source_path),
        simulate=args.simulate,
        warn_pre_release=False,
        settings_file=ScopeSession.user_settings_file(source_path),
    )
    try:
        try:
            server = build_server(session, shut_down=session.shutdown)
        except RestServerNotStartedError as e:
            logger.error(f'[REST     ] {e}')
            print(e, file=sys.stderr)
            session.shutdown()
            return 1
        # A failed home is reported and the motion ends; the server still
        # serves, so a client can home again.
        session.run_application_session()
        session.load_plugins()
        session.enable_metrics()
        reach = server.reach
        logger.info(f'[REST     ] Serving {reach}')
        print(f'LumaViewPro REST server: {reach.url} ({reach}; reference: /docs)', flush=True)
        server.run()
        if not server.close_begun:
            # It stopped without its close: it did not start serving, or was stopped before it did.
            session.shutdown()
            return 1
        if server.close_failed is not None:
            logger.error(f'[REST     ] The session did not close cleanly: {server.close_failed!r}')
            return 1
        return 0
    except BaseException:
        session.shutdown()
        raise
    finally:
        instance_lock.close()


if __name__ == '__main__':
    sys.exit(main())
