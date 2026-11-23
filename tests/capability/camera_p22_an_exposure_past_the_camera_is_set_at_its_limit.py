"""Camera P22: an exposure typed past the camera's maximum is set at it, stored so, told once, and saved so.

`ScopeSession.set_channel_settings` holds an exposure past the attached
camera's range to the limit, stores the limit, and reports the clamp in one
`ValueClampedNotice`; a still of the channel is then taken and saved at the
stored exposure, and its file says so (`read_file_record`). The case met:
3000 ms typed into a bare daA3840 brought up as the Lumi model (2026-11-16),
whose 1 s maximum is the camera's own, a figure the simulator does not hold.

Steps, every one through the Session, on the channel bring-up shows (the
simulator's is the Lumi model's one channel): 3000 ms written to its
exposure; the answer, the stored value and the notices read; one still saved
and its record read; its exposure put back as it was and the LEDs off.

    python tests/capability/camera_p22_an_exposure_past_the_camera_is_set_at_its_limit.py             # simulator
    python tests/capability/camera_p22_an_exposure_past_the_camera_is_set_at_its_limit.py --hardware  # the connected scope

Nothing moves beyond the bring-up home.
"""

import sys
import traceback

import harness
from _lumascope.modules.scope_parts import Board
from harness import HARDWARE, check, figure, hardware_session, report

from _lumascope.modules.exceptions import ValueClampedNotice
from _lumascope.modules.image_utils import read_file_record
from _lumascope.modules.notification_center import Severity, notifications

TYPED_MS = 3000.0
# A camera writes exposure in its own steps; the file carries what the camera
# reported, rounded.
EXPOSURE_TOLERANCE_MS = 0.05


def body(session):
    scope = session.scope
    maximum = scope.camera.setting_limits.exposure_max_ms
    figure('camera_model', scope.capabilities.camera_model)
    figure('exposure_max_ms', maximum)
    check(
        'the camera publishes an exposure maximum below the typed value',
        maximum is not None and maximum < TYPED_MS,
        str(maximum),
    )
    channel = session.shown_channel.channel
    before = session.settings_snapshot[channel]['exposure_ms']
    figure('channel', channel)
    figure('exposure_before_ms', before)

    told = []
    notifications.add_listener(told.append, min_severity=Severity.DEBUG)
    try:
        answered = session.set_channel_settings(channel, exposure_ms=TYPED_MS).exposure_ms
        stored = session.settings_snapshot[channel]['exposure_ms']
        clamps = [
            post for post in told if getattr(post, 'reason', None) == ValueClampedNotice.reason
        ]
        figure('answered_ms', answered)
        figure('stored_ms', stored)
        figure('clamp_notices', [post.message for post in clamps])
        check(
            'the answer is the camera maximum', answered == maximum, f'{answered} against {maximum}'
        )
        check(
            'the stored exposure is the camera maximum',
            stored == maximum,
            f'{stored} against {maximum}',
        )
        check('the clamp is told once', len(clamps) == 1, f'{len(clamps)} clamp notices')

        stills = session.manual_capture.start_save_still().result(timeout=60)
        path = stills[0].file
        record = read_file_record(path)
        figure('still', str(path))
        check('the file carries its record', record is not None, str(path))
        if record is not None:
            in_file = record.get('exposure_time_ms')
            figure('file_exposure_ms', in_file)
            check(
                'the file names the channel',
                record.get('channel') == channel,
                str(record.get('channel')),
            )
            check(
                'the file says the camera maximum, not the typed value',
                in_file is not None and abs(in_file - maximum) <= EXPOSURE_TOLERANCE_MS,
                f'file {in_file}, maximum {maximum}',
            )
    finally:
        notifications.remove_listener(told.append)
        session.set_channel_settings(channel, exposure_ms=before)
        scope.illumination.leds_off()


def main():
    if HARDWARE:
        with hardware_session(needs=(Board.CAMERA,)) as (session, _runner):
            body(session)
        return
    session, _live = harness.make_session('camera_p22', microscope='Lumi')
    try:
        body(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
