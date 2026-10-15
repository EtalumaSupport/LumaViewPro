"""Camera P04: the camera's transport layer, read through the diagnostic probe -- its register-access timeouts and retries.

A camera setting write waits on the camera lane for `_CAMERA_WRITE_TIMEOUT_S`
(5 s), which since W2d part F covers a write and its one resend. The
transport layer's own write timeout and retry count say how long one
attempt can take; W2d's bench row B1 sizes the bound from them. Their node
names differ by camera and pylon release, so the probe reads every node the
transport layer has, through `diagnostics.run_pylon_diagnostic_probe`, and
names those that look like a timeout or a retry count.

    python tests/capability/camera_p04_transport_layer.py             # simulator: no Pylon camera
    python tests/capability/camera_p04_transport_layer.py --hardware  # the connected scope

Reads only; the probe drains the camera's error queue, which is its one
write. Nothing moves beyond the bring-up home.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

SAMPLE_S = 1.0


def _read(session):
    snapshot = session.scope.diagnostics.run_pylon_diagnostic_probe(duration_s=SAMPLE_S)
    if not HARDWARE:
        check(
            'a camera with no Pylon snapshot answers that it has none',
            snapshot.get('supported') is False,
            snapshot,
        )
        return
    figure('camera', snapshot.get('camera'))
    figure('pylon', snapshot.get('host', {}).get('camera_sdk'))
    layer = snapshot.get('transport_layer', {})
    figure('transport layer nodes', layer)
    figure(
        'timeouts and retries',
        {k: v for k, v in layer.items() if 'timeout' in k.lower() or 'retr' in k.lower()},
    )
    figure('snapshot errors', snapshot.get('errors'))
    figure('written to', snapshot.get('output_path'))
    check('the transport layer was listed', bool(layer), snapshot.get('errors'))


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _read(session)
    else:
        session, _live = make_session('probe_transport_layer')
        try:
            _read(session)
        finally:
            session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
