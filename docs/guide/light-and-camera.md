# Light and camera

Channels, the LEDs, the shown channel, and the camera's settings.

## Illumination

**Channels.** A channel is a way of imaging the sample: brightfield (`BF`),
phase contrast (`PC`), darkfield (`DF`), a fluorescence channel named for its
excitation (`Ex405`, `Ex488`, `Ex550`, `Ex589`, `Ex640`), or luminescence
(`Lumi`), which lights nothing. Which channels this scope has, and which LEDs
each drives, is `scope.channel_identity`. A channel is always named, never
numbered: a number where a channel is taken is refused.

```python
for channel in scope.channel_identity.channels:
    print(channel.key_name, channel.label, channel.led_addresses)
```

A file written before fluorescence channels were named for their excitation
(`Blue`, `Green`, `Red`) is read by the LED each old name lit; a caller's old
name is refused, as any unknown one is (`'channel_unknown'`, or `'not_a_setting'`
in a settings path).

**LEDs.** `led_on(channel, illumination_ma)` lights a channel's LEDs and
answers the current commanded ([State and settings](concepts.md#state-and-settings));
`led_off(channel)` and `leds_off()` turn them off. A current above zero but
below one of the board's steps is raised to one step, and one above the board's
ceiling, `scope.capabilities.led_max_ma`, is refused
(`'illumination_out_of_range'`). A channel this scope lacks, or one with no LED
(`Lumi`), is refused `'not_on_this_scope'`; one the release does not know,
`'channel_unknown'`. A write the LED board does not answer raises
`BoardNoReplyError`, and the LED then reads `LedState.UNKNOWN` until it is
written again.

```python
scope.illumination.led_on('Ex405', 50)
print(scope.illumination.get_led_states())  # one LedReading per board address
scope.illumination.leds_off()
```

**The API holds the truth.** Which LED is lit, at what current, and who holds
it is held by the illumination API; the drivers hold nothing. A run, an
autofocus, a diagnostic or a home holds the LEDs for its length, and a write
from anyone else is refused `'exclusive_activity_running'`. While a recording
runs, lighting an LED that is dark or at 0 mA is refused the same way;
changing the current of an LED lit above 0 mA, and turning one off, are open. Work that borrows the
light and gives it back reads it first and restores it after
(`get_led_states`, then `restore_led_state` with what it answered).

**The shown channel.** LumaViewPro shows one channel at a time, and so does
the session: `session.show_channel(channel)` puts that channel's stored gain,
exposure and auto-gain on the camera (with no camera connected the channel is
still shown and the camera half is skipped), and `session.light_shown_channel(lit)`
lights or darkens it at its stored current. This is the one way a channel's
settings reach the camera; a run and a go-to-step move it too. Its `lit` is
what the LEDs read, whoever lit or darkened them: after a raw
`illumination.leds_off()` it is dark, and the next channel shown is dark too.
Each move of the shown channel and each LED change is an event
([Events and listeners](python.md#events-and-listeners);
[The event stream](rest.md#the-event-stream)).

```python
print(session.show_channel('BF'))           # ShownChannel(channel='BF', lit=False)
print(session.light_shown_channel(True))    # lit at BF.illumination_ma
session.light_shown_channel(False)
```

## Camera

**Where things are.** The camera's settings are `scope.camera`; taking frames
is `scope.imaging`. One member writes each setting, and the settings LumaViewPro
keeps (binning, frame size, image mode) are saved by that member, whoever calls
it.

**A write applies or raises.** Every camera write either took effect or raises:
a value outside the camera's declared range is refused before it reaches the
camera (`CameraSettingOutOfRangeError`, its `limits` the range), and a value
the camera refused or did not answer is the fault `CameraSettingRejected`
(`'camera_rejected'`, `'camera_no_answer'`). Each setter answers what the camera
now runs, which may differ from what was asked.

```python
gain = scope.camera.set_gain_db(3.3)
exposure = scope.camera.set_exposure_ms(5.0)
print(gain, exposure)
print(scope.capabilities.camera_limits)     # the declared ranges, read once at connect
```

```python raises=CameraSettingOutOfRangeError:gain_db_out_of_range
scope.camera.set_gain_db(999.0)             # gain_db_out_of_range
```

**Held and fresh reads.** Each setting has two reads. The held read, a
property named for the setting (`gain_db`, `exposure_ms`, `binning`,
`frame_size`, ...), answers what the camera last took or reported, with no call
to the camera, and is None whenever that is unknown. The fresh read, `read_*`
(`read_gain_db`, `read_exposure_ms`, ...), asks the camera now and raises
`CameraReadFailedError` when it does not answer. It asks from the camera's
lane, after whatever camera command is queued there, so it never lands inside
a write; a run, a home or a control session does not refuse it.

```python
print(scope.camera.gain_db, scope.camera.read_gain_db())
```

**A raw write and a channel's settings.** The camera's setters are raw writes:
`set_gain_db`, `set_exposure_ms`, `set_auto_gain` and `apply` change the camera
now, and the written values stand -- a still takes its frame with them, its
record's `held_channel` None -- until a channel's settings are put back:
`show_channel`, a run's step, `go_to_step`, or a write to the shown channel's
gain or exposure. A channel's own gain, exposure and auto-gain are its stored
settings, which a capture of that channel uses:
`session.update_settings('BF.gain_db', ...)`,
`session.update_settings('BF.exposure_ms', ...)` and
`session.set_channel_auto_gain('BF', ...)`. A script that images a channel
sets the channel's settings; a raw write is for work on the camera itself.

**Auto gain and exposure.** `set_auto_gain(state, settings)` turns the
camera's auto-gain on or off with its target settings, and `lock_auto_gain()`
fixes the gain and exposure it reached. `set_auto_exposure(mode)` takes a
`CameraAutoMode` (`OFF`, `CONTINUOUS`). On a camera without the feature the
reads answer `OFF`, and turning it on is refused
`CameraSettingUnsupportedError`.

**Black level.** `set_black_level(level)` writes the camera's black level,
`read_black_level()` reads it fresh, and `get_black_level_range()` answers its
range; every saved file records it.

**Geometry and image mode.** `set_binning(size)` and
`set_frame_size(width, height)` answer the `FrameSize` the camera delivers;
the frame size is floored to even sides and refused below
`camera_limits.frame_min` or above the largest frame at the binning held:
`camera_limits.frame_max`, the unbinned maximum, divided by the binning.
`frame_at_binning(size)` answers the frame `set_binning(size)` will ask for,
without applying it. `set_image_mode(mode)` chooses the
`ImageMode`: 8-bit, or 12-bit scientific, scaled or false-colour. A camera that
lacks a binning factor or a feature refuses it `CameraSettingUnsupportedError`.

```python
from lumascope import ImageMode

print(scope.camera.set_binning(2))
print(scope.camera.set_frame_size(960, 600))
scope.camera.set_binning(1)
scope.camera.set_image_mode(ImageMode.EIGHT_BIT)
```

A camera setting changed while a run, a diagnostic or a home holds the scope is
refused (`'exclusive_activity_running'`). During a recording only what would
falsify its frames -- binning, frame size, image mode -- is refused
([Who may act](concepts.md#who-may-act) has the whole list).

**Streaming.** The camera streams from bring-up on. `scope.imaging.disable_streaming()`
stops its feed and `enable_streaming()` starts it again; `is_streaming()`
answers which. While the feed is stopped, a still, a saved still, a run's start
and a go-to are refused (`'camera_not_streaming'`). The camera's other members -- the conversion gain,
line-noise reduction, the longest exposure, the frame rate it reports -- are in
the [reference](../api-reference.md#CameraAPI).
