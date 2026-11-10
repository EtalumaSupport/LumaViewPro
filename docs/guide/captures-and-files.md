# Captures and files

Frame validity, which channel a still is, saved stills, and what a saved file holds.

## Frames and captures

**Frame validity.** After a change -- an LED lit, an exposure set, the stage
moved -- the camera keeps delivering frames that were exposed before the change
took effect. A capture waits until a frame counts for every recorded change,
by counting frames: a frame counts once the change's source's count of frames
has arrived since the change (a move's since its stop): the count the camera's
timing record measured -- for a Z or turret move the largest it holds over the
models the camera ships in, while an XY move keeps its shipped count -- else the
shipped one. `scope.capabilities.frame_rules` answers each source's count, and
`scope.imaging.changes_in_force()` whether a frame arriving now would count. A
capture never infers validity from pixel content, though a gain or exposure
frame whose own chunk data disagrees with the change is not counted. When the
scope keeps changing past the capture's deadline it raises
`CaptureFailedError` (`'deadline_expired'`); its other reasons -- no frame, no
fresh frame, a rejected chunk, a record it could not read -- are
[its reference's](../api-reference.md#CaptureFailedError). A recording, manual or a run's video step, waits the
same before its first frame, so it keeps no frame made before a change was in
force; a manual start whose change does not come into force within a capture's
limit is refused, `RecordingRefusedError` (`'changes_not_in_force'`).

**Which channel a still is.** A still is named for the shown channel, and the
camera holds that channel's stored gain, exposure and auto-gain for the frame
and after it, whatever came before: a run, a go-to. It is lit by the shown
channel's LED alone: with another LED lit through `scope.illumination`, the
still is refused, `HardwareCommandRefusedError` `'other_led_lit'`, before
anything changes, as a recording's start is. To take a still of another
channel, show it. After a raw camera write (`scope.camera.set_gain_db` and the
other setters) the written values stand, and the frame's record says so: its
`held_channel` is None.

**Dark and saturated frames are returned, not refused.** With an LED lit, a
frame with no lit pixel is retried until `timeout_s` and then returned, its
`record.dark` True; with nothing lit a dark frame is expected and is not judged
(`dark` None). With `all_ones_check=True` a frame at least 98% saturated is
taken again once and then returned, its `saturated_fraction` measured. Read the
record to tell; never re-measure the pixels.

**A saved still.** `session.manual_capture.capture(...)` takes the still
LumaViewPro's Capture button takes, and saves it: the summing and the format
come from the settings. It returns an `Operation` of a `SavedStill` per file
written, each its `file` and the record of it, `record` (a `FrameRecord`, as
`capture_still` answers it). The file is
`<live folder>/Manual/live_<well>_<channel>_<number>.<ext>`, the well left out
where the position has none and the number counting up from `000001`; with
`separate_folder_per_channel` set it is in a folder per channel under
`Manual/`. The live folder is `live_folder` in the settings, set by
`session.set_live_folder(folder)`, which creates it. `false_color_on` is the
false-colour toggle of LumaViewPro's channel controls: on, the file carries
the channel's colour for display ([Saved files](#saved-files)); off, it is
grey. It changes how the file is shown, never what it records.

```python
scope.motion.move_to_well('B2')             # over the simulator's specimen
session.show_channel('BF')
scope.illumination.led_on('BF', 5)
saving = session.manual_capture.capture(false_color_on=False)
saved = saving.result(timeout=30)
print(saved[0].file, saved[0].channel, saved[0].statistics.maximum)
assert saved[0].well_label == 'B2' and saved[0].statistics.maximum > 0   # at B2, and lit
scope.illumination.leds_off()
```

**A recording.** `session.manual_recording.start()` starts the recording
LumaViewPro's Record button starts and returns its `Operation`: `stop()` ends
it, and `result()` answers a `RecordingFinished`, its `end_reason` and
`save_wait_s`. It keeps at most `video.max_fps` frames a second (0, shipped, is
every frame) for at most `video.max_duration_s`, as numbered TIFFs when
`video_as_frames` is set and as an MP4 otherwise ([Saved files](#saved-files));
`is_recording` and `elapsed_s` answer where it stands. A start that cannot
record is refused `RecordingRefusedError`.

```python
recording = session.manual_recording.start()
recording.stop()
finished = recording.result(timeout=60)
print(finished.end_reason, finished.save_wait_s)   # why it ended, and how long saving ran on
```

A bare image carries no record. A file with the instrument's record in it is
saved by the API that took the frame -- a still, a run, a recording -- and
never by your program from the array ([Saved files](#saved-files)).

## Saved files

**What a still holds.** A still is saved under the live folder in the format
the settings name (`OutputFormat`: TIFF, OME-TIFF or JPG; a run may instead
build an OME-TIFF hyperstack per well). A TIFF or OME-TIFF keeps mono pixels.
With false colour off they are grey and carry no colour. With it on, the
channel's colour (grey for BF, PC and DF, so they stay grey) is carried beside them for display -- an 8-bit file as a
palette, a full-depth one as an ImageJ lookup table or OME `Channel.Color` --
except in `FALSE_COLOR_RGB`, which bakes it into three-channel pixels. A JPG
is the image as it is shown: 8-bit, grey or its channel's colour baked into
its pixels, and no record in the file; a manual still's record is the
`SavedStill` its capture answered.

**The image mode decides the depth and the encoding.** `ImageMode.EIGHT_BIT`
captures and saves 8-bit mono, and saves a sum of frames as it is shown,
saturating at one frame's white. The full-depth modes capture what the camera
has -- 12 bits where it delivers 12 -- and save it right-aligned (`SCIENTIFIC`:
a 12-bit frame is 0 to 4095 in a 16-bit container), shifted to fill the
container (`SCALED`), or, with false colour on, as three-channel RGB in the
channel's colour (`FALSE_COLOR_RGB`). The file states the depth its values
reach, its record's `significant_bits` -- 12 for a 12-bit `SCIENTIFIC` frame,
16 for a `SCALED` one -- so divide by `(1 << significant_bits) - 1` to scale
it. Every TIFF also carries it in private tag 65123, and an OME-TIFF in its
schema's `SignificantBits`, so a reader that knows neither LVP's record nor
OME still finds it. A still's or a run's image also carries private tag 65124,
the depth one frame's white is at: for one frame the same as 65123 (8 and 8 for
an 8-bit still), and for a full-depth sum below it -- the sum states the bits
it can reach, 14 for four 12-bit frames, and 65124 is 12. Every sum states
`frames_summed` (`FramesSummed`), how many it holds.

**The record in the file.** Each TIFF and OME-TIFF carries what the instrument
was when the frame was taken: the instrument (model, serial, camera), the
objective and pixel size, the plate and well, the position and whether the
stage was moving, each lit channel's current and when it was read, the
excitation, the exposure, gain, black level and binning, whether the frame was
dark or saturated, the timestamps and where they came from, whether the camera
held the channel's settings, and the frame-validity judgments. An OME-TIFF keeps the exposure and position in
its OME `Plane` and the rest in a `MapAnnotation` (namespace `etaluma.com/lvp`);
a plain TIFF carries the same keys as JSON, its `Plane` block, in its second
`ImageDescription`. Those are the typed record LVP reads back. Every file also
carries the record as a person reads it, plain `key = value` pairs (a plain
TIFF's first `ImageDescription`, an ImageJ TIFF's `Info`), which an image
viewer lists with the file's metadata (in Fiji, the Bio-Formats importer's
*Display metadata*). Its keys are the file's own, not `FrameRecord`'s fields:
`exposure_time_ms` where the record has `exposure_ms`, `camera_held_channel`
where it has `held_channel`, and `illumination_ma` the one current, with
`illumination_by_channel_ma.<channel>` each lit channel's, where the record
keys its currents by LED address. The channel and its light are `channel`,
`camera_held_channel`, `illumination_ma` and `dark`; `filterset` is the scope's
filter set, whatever channel was imaged. Every key, its unit and what a sum
does to it is the reference's [saved-file
facts](../api-reference.md#saved-file-facts). A fact the frame did not know is
absent, never a stand-in. A few values name a type of the reference:
`illumination_timing` is a [`LightTiming`](../api-reference.md#LightTiming),
when the currents were read; each `frame_validity.<n>` is a
[`Judgment`](../api-reference.md#Judgment) of the frame against one change --
`counts`, `frames_after`, `after_write_ms`, `after_in_force_ms` -- with its
[`Change`](../api-reference.md#Change)'s `source` and `skip`, the frames it
needed, and `counted_by`, each counted frame's number and the
[`Counter`](../api-reference.md#Counter) that counted it (`287:drain`).

The API has no reader of a saved file's record: read it with any TIFF reader,
or keep the `SavedStill` or the run's record the capture answered. In Python:

```python
import tifffile

session.show_channel('BF')
scope.illumination.led_on('BF', 5)
path = session.manual_capture.capture(false_color_on=False).result(timeout=30)[0].file
scope.illumination.leds_off()

with tifffile.TiffFile(path) as tif:    # a plain TIFF's first description is the view
    view = dict(
        line.split(' = ', 1) for line in tif.pages[0].description.splitlines() if ' = ' in line
    )
print(view['channel'], view['illumination_ma'], view['dark'])
assert view['channel'] == 'BF' and view['dark'] == 'false'
```

**Derived files.** A z-projection, a stitch, a composite or a video states a
fact only when every input states the same: a composite of several channels
states no current, and a sum of slices no `FramesSummed`. A run's hyperstack is
an OME-TIFF per well and scan, its channels coloured by OME `Channel.Color`; in
Fiji open it with `Plugins > Bio-Formats > Importer`, colour mode Composite.

**Recordings.** A recording is saved as numbered TIFFs, each frame with its
own record (position, whether the stage was moving, the channel and its
current), or as an MP4 played at the frames' own times; either way a
`recording_manifest.json` holds what was measured: the frames delivered, their
timestamps and the rate. A rate is a limit: with one set, the manifest's
`frame_budget` is the frames the limit and the duration allow, and a camera
slower than the limit ends at the duration with fewer (`frames_selected`).
No frame is dropped for a slow disk, so saving can run on after the recording
ends; a frame lost to a write error, or one that arrived without the camera's
timestamp, is counted and told. How long saving ran on is the manifest's
`save_wait_s`, a manual recording's answer (`RecordingFinished.save_wait_s`) and a run's (`RunOutcome.video_save_wait_s`),
and a wait over 10 s is told once saving finishes. Frames saved as files are
also built into a hyperstack of their channel when the run format is the
hyperstack.
