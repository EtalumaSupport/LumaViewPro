"""Camera P15: high conversion gain and line-noise reduction store and apply, on and off.

Each toggle's member (``CameraAPI.set_high_conversion_gain``,
``set_line_noise_reduction``) writes the camera and, once the camera took
it, saves the setting; the camera's held state (``high_conversion_gain``,
``line_noise_reduction``) and the session's stored setting then agree. No
driver reads either back, so on the camera the write's acceptance is the
read, and for conversion gain its effect too: a still at the same light is
brighter with high conversion gain on. Each toggle ends as it began.

On a camera without a mode (the capability probe found no node: the
simulated camera, and the LS850T's daA3840-45um, which has neither), on is
refused with its reason and nothing is sent, and off is its state.

    python tests/capability/camera_p15_the_camera_toggles_store_and_apply.py
    python tests/capability/camera_p15_the_camera_toggles_store_and_apply.py --hardware

The settings are written on the session in memory; the installation's
current.json is left as it was, which ``hardware_session()`` checks.
"""

import argparse
import sys

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--layer', default='BF')
    return parser.parse_args()


def _toggle(session, name, setter, held, stored_path):
    camera = session.scope.camera
    start = session.get_settings_snapshot()['camera'][stored_path]
    figure(f'{name}.stored_at_start', start)
    for enabled in (True, False, True, False):
        setter(enabled)
        stored = session.get_settings_snapshot()['camera'][stored_path]
        check(
            f'{name}: {"on" if enabled else "off"} is held and stored',
            getattr(camera, held) is enabled and stored is enabled,
            f'held {getattr(camera, held)!r}, stored {stored!r}',
        )
    setter(bool(start))


def _absent(session, name, setter, held, reason):
    """A camera without the mode: on is refused with its reason, off is its state."""
    from modules.exceptions import CameraSettingUnsupportedError

    try:
        setter(True)
        check(f'{name}: on is refused on a camera without it', False, 'no raise')
    except CameraSettingUnsupportedError as refused:
        check(
            f'{name}: on is refused on a camera without it',
            refused.reason == reason,
            refused.reason,
        )
    setter(False)
    check(
        f'{name}: off is its state',
        getattr(session.scope.camera, held) is False,
        repr(getattr(session.scope.camera, held)),
    )


def _mean(session, layer):
    return float(session.capture_still(layer, force_to_8bit=False, accept_dark=True).image.mean())


def main():
    args = _arguments()
    live = probe_dir('camera_p15')
    session_cm = hardware_session() if HARDWARE else headless_session(live, acquiring=(args.layer,))
    with session_cm as (session, _runner):
        caps = session.scope.capabilities
        camera = session.scope.camera
        figure('camera', caps.camera_model)
        has_cg = caps.camera_supports_conversion_gain_mode
        has_ln = caps.camera_supports_line_noise_reduction
        figure('supports.conversion_gain_mode', has_cg)
        figure('supports.line_noise_reduction', has_ln)
        # Which side each toggle runs is the camera's: a camera with the mode
        # stores and applies it, one without refuses on and keeps off.
        if has_cg:
            _toggle(
                session,
                'high_conversion_gain',
                camera.set_high_conversion_gain,
                'high_conversion_gain',
                'high_conversion_gain',
            )
            session.show_channel(args.layer)
            camera.set_high_conversion_gain(False)
            low = _mean(session, args.layer)
            camera.set_high_conversion_gain(True)
            high = _mean(session, args.layer)
            camera.set_high_conversion_gain(False)
            figure('still_mean.low_conversion_gain', round(low, 2))
            figure('still_mean.high_conversion_gain', round(high, 2))
            check(
                'a still is brighter with high conversion gain on',
                high > low * 1.1,
                f'{low:.2f} -> {high:.2f}',
            )
        if has_ln:
            _toggle(
                session,
                'line_noise_reduction',
                camera.set_line_noise_reduction,
                'line_noise_reduction',
                'line_noise_reduction',
            )
        if not has_cg:
            _absent(
                session,
                'high_conversion_gain',
                camera.set_high_conversion_gain,
                'high_conversion_gain',
                'conversion_gain_mode_unsupported',
            )
        if not has_ln:
            _absent(
                session,
                'line_noise_reduction',
                camera.set_line_noise_reduction,
                'line_noise_reduction',
                'line_noise_reduction_unsupported',
            )
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
