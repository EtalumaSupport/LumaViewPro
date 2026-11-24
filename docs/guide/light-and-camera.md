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
(`Blue`, `Green`, `Red`) is read by the LED each old name lit. A caller's old
name is refused, as any unknown one is (`'channel_unknown'`, or `'not_a_setting'`
in a settings path).

**LEDs.** `led_on(channel, illumination_ma)` lights a channel's LEDs and
answers the current commanded ([Intent and what was delivered](concepts.md#intent-and-what-was-delivered)).
`led_off(channel)` and `leds_off()` turn them off. A current above zero but
below one of the board's steps is raised to one step, and one above the board's
ceiling, `scope.capabilities.led_max_ma`, is lit at the ceiling and reported
(`ValueClampedNotice`). One below zero is refused
(`'illumination_out_of_range'`). A channel this scope lacks, or one with no LED
(`Lumi`), is refused `'not_on_this_scope'`. One the release does not know is
refused `'channel_unknown'`. A write the LED board does not answer raises
`BoardNoReplyError`, and the LED then reads `LedState.UNKNOWN` until it is
written again.

```python
scope.illumination.led_on('Ex405', 50)
print(scope.illumination.led_states)  # one LedReading per board address
scope.illumination.leds_off()
```

Over REST: `POST /api/v1/scope/illumination/led_on` with
`{"channel": "BF", "illumination_ma": 5.0}` answers the current commanded,
and `POST /api/v1/scope/illumination/leds_off` with `{}` turns every LED off.

**The API keeps the truth.** The illumination API keeps which LED is lit, at
what current, and who holds it. The drivers keep nothing. A run, an
autofocus, a diagnostic or a home holds the LEDs for its length, and a write
from anyone else is refused `'exclusive_activity_running'`. While a recording
runs, lighting an LED that is dark or at 0 mA is refused the same way.
Changing the current of an LED lit above 0 mA, and turning one off, are open. Work that borrows the
light and gives it back reads it first and restores it after
(`led_states`, then `restore_led_state` with what it answered).

**The shown channel.** LumaViewPro shows one channel at a time, and so does
the session. `session.show_channel(channel)` puts that channel's stored gain,
exposure and auto-gain on the camera. With no camera connected, the channel is
still shown and the camera half is skipped. `session.light_shown_channel(lit)`
lights or darkens it at its stored current. This is the one way a channel's
settings reach the camera. A run and a go-to-step move it too. Its `lit` is
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

Over REST: `POST /api/v1/show_channel` with `{"channel": "BF"}`, and
`POST /api/v1/light_shown_channel` with `{"lit": true}`.

## Camera

**Where things are.** The camera's settings are `scope.camera`, and taking
frames is `scope.imaging`. One member writes each setting, and the settings LumaViewPro
keeps (binning, frame size, image mode) are saved by that member, whoever calls
it.

**A write applies or raises.** Every camera write either took effect or raises.
A gain, exposure or black level past the camera's declared range is set at the
range's end and reported once, in a `ValueClampedNotice` on the event stream.
The range for gain and exposure is `scope.camera.setting_limits`, read again
after each change of the camera's readout, since a pixel format or a binning
moves it. A value no range makes sense of -- not a finite number, an exposure
of 0 or below, a gain or black level below 0 -- is refused
(`ArgumentRefusedError`).

A value the camera refused or did not answer, or a write whose read failed, is
the fault [`CameraSettingRejected`](../api-reference.md#CameraSettingRejected),
and its `reason` says which. One setting can take several node writes, and the
camera may take some of them before it refuses one. So a refusal's `in_effect`
is the value in effect on the camera for the setting, read after the refusal
(None when that read fails), and its words say so. Each setter answers what the
camera now runs, which may differ from what was asked. It answers None when the
camera took the write and its value could not be read back.

```python
gain = scope.camera.set_gain_db(3.3)
exposure = scope.camera.set_exposure_ms(5.0)
print(gain, exposure)
print(scope.camera.setting_limits)          # the gain and exposure range in force
print(scope.capabilities.camera_limits)     # the frame limits, read once at connect
```

```python
gain = scope.camera.set_gain_db(999.0)      # past the range: set at its end, and told
assert gain == scope.camera.setting_limits.gain_max_db
```

```python raises=ArgumentRefusedError:wrong_kind
scope.camera.set_exposure_ms(0.0)           # wrong_kind: an exposure takes a number above 0
```

Over REST a camera setter is a `PUT` on its read's URL
([Routes](rest.md#routes)): `PUT /api/v1/scope/camera/gain_db` with
`{"gain_db": 3.3}` answers the gain now in effect, and a channel's own
settings are `PUT /api/v1/channel_settings`.

**Held and fresh reads.** Most settings have two reads. The held read, a
property named for the setting (`gain_db`, `exposure_ms`, `binning`,
`frame_size`, ...), answers what the camera last took or reported, with no call
to the camera, and is None whenever that is unknown. The fresh read, `read_*`
(`read_gain_db`, `read_exposure_ms`, ...), asks the camera now and raises
`CameraReadFailedError` when it does not answer. It asks from the camera's
lane, after whatever camera command is queued there, so it never lands inside
a write. A run, a home or a control session does not refuse it.

```python
print(scope.camera.gain_db, scope.camera.read_gain_db())
```

**A raw write and a channel's settings.** The camera's setters are raw writes:
`set_gain_db`, `set_exposure_ms`, `set_auto_gain` and `apply` change the camera
now. The written values stand until a channel's settings are put back:
`show_channel`, a run's step, `go_to_step`, or a write to the shown channel's
gain or exposure. Until then a still takes its frame with them, its record's
`held_channel` None. A channel's own gain, exposure and auto-gain are its stored
settings, which a capture of that channel uses:
`session.set_channel_settings('BF', gain_db=..., exposure_ms=..., auto_gain=...)`. A script that images a channel
sets the channel's settings. A raw write is for work on the camera itself.

**Auto gain and exposure.** `set_auto_gain(mode, settings)` and
`set_auto_exposure(mode)` take a `CameraAutoMode` (`OFF`, `CONTINUOUS`), the
closed set their reads answer. Auto-gain is armed with an `AutoGainSettings`,
its target and the bounds it settles in, given with `CONTINUOUS` and only
with it. `lock_auto_gain()` fixes the gain and exposure it reached. On a
camera without the feature the reads answer `OFF`, and turning it on is
refused `CameraSettingUnsupportedError`.

```python
from lumascope import AutoGainSettings, CameraAutoMode

scope.camera.set_auto_gain(
    CameraAutoMode.CONTINUOUS,
    AutoGainSettings(target_brightness=0.3, min_gain_db=0.0, max_gain_db=20.0),
)
lock = scope.camera.lock_auto_gain()
scope.camera.set_auto_gain(CameraAutoMode.OFF)
```

**Black level.** `set_black_level(level)` writes the camera's black level,
`read_black_level()` reads it fresh, and `read_black_level_range()` answers its
range. Every saved file records it.

**Geometry and image mode.** `set_binning(size)` and
`set_frame_size(width, height)` answer the `FrameSize` the camera delivers.
A side below `camera_limits.frame_min` (the smallest frame the camera
delivers, 2 x 2 on every camera) is set at that limit and reported
(`'value_clamped'`). So is a side above the largest frame at the binning in
effect (`camera_limits.frame_max`, the unbinned maximum, divided by the
binning). The frame is floored to even sides, and a side of 0 or below is
refused `'wrong_kind'`.
`frame_at_binning(size)` answers the `FrameSize` `set_binning(size)` will ask
for, without applying it. `set_image_mode(mode)` chooses the
`ImageMode`: 8-bit, or 12-bit scientific, scaled or false-color. A camera that
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
([The holders](concepts.md#the-holders) has the whole list).

**Streaming.** The camera streams from bring-up on. `scope.imaging.disable_streaming()`
stops its feed and `enable_streaming()` starts it again. `read_streaming()`
answers which, and raises `CameraReadFailedError` when the camera does not
answer. The session's `status` then carries `camera_streaming` as `null`. While the feed is stopped, a still, a saved still, a run's start
and a go-to are refused (`'camera_not_streaming'`). The camera's other members -- the conversion gain,
line-noise reduction, the longest exposure, the frame rate it reports -- are in
the [reference](../api-reference.md#CameraAPI).
