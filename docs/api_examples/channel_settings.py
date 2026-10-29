#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Channels, each with its own current, exposure and gain, on the simulator.

A channel's current, exposure and gain are settings, stored per channel:
written by path, they reach the camera when the channel is shown, and a
still of the channel is taken with them. This saves a still of brightfield
and of a fluorescence channel that way, then one still after writing the
camera directly, which the still's record tells apart. Prints what each
record says.

Run it in a LumaViewPro source folder (how, step by step: README.md here):

    python3 docs/api_examples/channel_settings.py

The same stills over REST are ``channel_settings.sh``.
"""

import pathlib
import sys

# The installation folder, importable when the script is run by its path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from modules.scope_session import ScopeSession

# Each channel's current (mA), exposure (ms) and gain (dB): each lights the
# simulator's specimen without clipping.
CHANNELS = {
    'BF': (5, 2.0, 1.0),
    'Ex488': (300, 100.0, 10.0),
}

# A raw camera write: the camera runs these until a channel is shown again.
RAW_EXPOSURE_MS = 1.5
RAW_GAIN_DB = 3.0


def report(still):
    frame = still.frame
    print(f'Saved {still.file.name}')
    print(f'  channel {still.channel}, camera held {frame.held_channel}')
    print(f'  {frame.exposure_ms} ms, {frame.gain_db} dB')
    print(f'  current of each LED lit, by address: {dict(frame.illumination_ma)} mA')
    print(f'  brightest pixel {still.statistics.maximum}')


def main():
    # No settings_file is passed, so these writes are not saved to the
    # user's configuration.
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        session.run_application_session()
        scope = session.scope
        scope.motion.move_to_well('B2')

        stored = {}
        for channel, (current_ma, exposure_ms, gain_db) in CHANNELS.items():
            # A write outside a setting's range is refused, never clamped.
            session.update_settings(f'{channel}.illumination_ma', current_ma)
            session.update_settings(f'{channel}.exposure_ms', exposure_ms)
            session.update_settings(f'{channel}.gain_db', gain_db)

            # Showing the channel puts its stored exposure and gain on the
            # camera; lighting it lights it at its stored current.
            session.show_channel(channel)
            session.light_shown_channel(True)
            saving = session.manual_capture.capture(false_color_on=False)
            stored[channel] = saving.result(timeout=30)[0]
            session.light_shown_channel(False)
            report(stored[channel])

        # A raw write stands for the next still, whose record says the camera
        # held no channel's settings.
        scope.camera.set_exposure_ms(RAW_EXPOSURE_MS)
        scope.camera.set_gain_db(RAW_GAIN_DB)
        scope.illumination.led_on('BF', 5)
        raw = session.manual_capture.capture(false_color_on=False).result(timeout=30)[0]
        scope.illumination.leds_off()
        report(raw)
        return stored, raw
    finally:
        session.shutdown()


if __name__ == '__main__':
    main()
