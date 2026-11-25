# Captures and files

A still, frame validity, which channel a still is, saved stills, and what a saved file holds.

## A still

`session.capture_still(channel)` takes one frame and returns a
`CapturedStill`: the read-only `image` and its `record`, a `FrameRecord` of
what the instrument was when the frame was taken. `channel` is the shown
channel (another is refused `'channel_not_shown'`), and another LED lit is
refused `'other_led_lit'`. `capture_still()` with no channel takes the frame
with the camera and the LEDs as they are. The image is 8-bit unless
`force_to_8bit=False` keeps the depth the image mode captures. `frames_summed`
sums 1 to 100 frames (one above 100 sums 100 and reports `'value_clamped'`).
`still.significant_bits` is the image's depth. The record's
`illumination_ma` is keyed by LED board address; `scope.channel_identity` maps
a channel to its addresses.

```python
scope.motion.move_to_well('B2')             # over the simulator's specimen
session.show_channel('BF')
session.light_shown_channel(True)           # at BF's stored current, 5 mA
still = session.capture_still('BF', all_ones_check=True)
record = still.record
print(still.image.shape, still.image.dtype)
print(record.held_channel, record.exposure_ms, record.gain_db, record.illumination_ma)
assert record.dark is False                 # lit, and light was seen
assert record.saturated_fraction < 0.98     # measured, because all_ones_check asked
```

Over REST: `POST /api/v1/still`, its body the member's arguments, answers the
still's record and then its pixels when the still is taken, never as a job
([The live view and files](rest.md#the-live-view-and-files)).

## Frames and captures

**Frame validity.** After a change -- an LED lit, an exposure set, the stage
moved -- the camera keeps delivering frames that were exposed before the change
took effect. A capture waits by counting frames. Each source of a change has a
count, and a frame counts for a change once that many frames have arrived
since the change (for a move, since its stop). The count is the largest the
camera's timing record measured for that source on any model the camera ships
in, else the shipped count. A Z or turret move takes the largest measured over
both, and an XY move keeps its shipped count. A capture waits until a frame
counts for every recorded change. `scope.capabilities.frame_rules` answers each source's count, and
`scope.imaging.changes_in_force()` whether a frame arriving now would count. A
capture never infers validity from pixel content, though a gain or exposure
frame whose own chunk data disagrees with the change is not counted. When the
scope keeps changing past the capture's deadline it raises
`CaptureFailedError` (`'deadline_expired'`). Its other reasons -- no frame, no
fresh frame, a rejected chunk, a record it could not read -- are
[its reference's](../api-reference.md#CaptureFailedError). A recording, manual
or a run's video step, waits the same before its first frame, so it keeps no
frame made before a change was in force. A manual start whose change does not come into force within a capture's
limit is refused, `RecordingRefusedError` (`'changes_not_in_force'`).

**Which channel a still is.** A still is named for the shown channel, and the
camera holds that channel's stored gain, exposure and auto-gain for the frame
and after it, whatever came before: a run, a go-to. It is lit by the shown
channel's LED alone. With another LED lit through `scope.illumination`, a saved
still and `capture_still(channel)` are refused before anything changes,
`HardwareCommandRefusedError` `'other_led_lit'`. A recording's start is
refused too, `RecordingRefusedError` `'other_led_lit'`. `capture_still()` with
no channel takes the LEDs as they are ([A still](#a-still)). To take a still of another
channel, show it. After a raw camera write (`scope.camera.set_gain_db` and the
other setters) the written values stand, and the frame's record says so: its
`held_channel` is None.

**Dark and saturated frames are returned, not refused.** With an LED lit, a
frame with no lit pixel is retried until `timeout_s` and then returned, its
`record.dark` True. With nothing lit, a dark frame is expected and is not judged
(`dark` None). With `all_ones_check=True` a frame at least 98% saturated is
taken again once and then returned, its `saturated_fraction` measured. Read the
record to tell; never re-measure the pixels.

**A saved still.** `session.manual_capture.start_save_still(...)` takes the still
LumaViewPro's Capture button takes, and saves it: the summing and the format
come from the settings. It returns an `Operation` of a `SavedStill` per file
written, each its `file` and the record of it, `record` (a `FrameRecord`, as
`capture_still` answers it). The file is
`<live folder>/Manual/live_<well>_<channel>_<number>.<ext>`, the well left out
where the position has none and the number counting up from `000001`. With
`separate_folder_per_channel` set, it is in a folder per channel under
`Manual/`. The live folder is `live_folder` in the settings, set by
`session.set_live_folder(folder)`, which creates it. The channel's stored
`false_color` setting (`<channel>.false_color`, the false-color toggle of
LumaViewPro's channel controls) decides how the file is rendered: on, it
carries the channel's color for display ([Saved files](#saved-files)); off,
it is gray. A manual recording follows the same setting. It changes how the
file is shown, never what it records.

```python
scope.motion.move_to_well('B2')             # over the simulator's specimen
session.show_channel('BF')
session.light_shown_channel(True)
saving = session.manual_capture.start_save_still()
saved = saving.result(timeout=30)
print(saved[0].file, saved[0].channel, saved[0].statistics.maximum)
assert saved[0].well_label == 'B2' and saved[0].statistics.maximum > 0   # at B2, and lit
session.light_shown_channel(False)
```

Over REST: `POST /api/v1/manual_capture/start_save_still`, answered `202` with
its job, whose `result` is the files written ([Jobs](rest.md#jobs)). A file
downloads by its name in the live folder, `GET /api/v1/files/<name>`
([The live view and files](rest.md#the-live-view-and-files)).

**A recording.** `session.manual_recording.start()` starts the recording
LumaViewPro's Record button starts and returns its `Operation`: `stop()` ends
it, and `result()` answers a `RecordingFinished`, its `end_reason` and
`save_wait_s`. It keeps at most `video.max_fps` frames a second (0, shipped, is
every frame) for at most `video.max_duration_s`. Its format is the one
`video_recording_format` names: mp4 (shipped), one MP4; frames, numbered
TIFFs; hyperstack, the TIFFs and then one OME-TIFF hyperstack of them beside
them. A protocol's video steps follow the same setting ([Saved files](#saved-files)).
`is_recording` and `elapsed_s` answer where it stands. A start that cannot
record is refused `RecordingRefusedError`.

```python
recording = session.manual_recording.start()
recording.stop()
finished = recording.result(timeout=60)
print(finished.end_reason, finished.save_wait_s)   # why it ended, and how long saving ran on
```

Over REST: `POST /api/v1/manual_recording/start`, answered `202` with its job.
`DELETE /api/v1/jobs/<id>` stops it, as `stop()` does, and the job's `result`
is the `RecordingFinished` ([Jobs](rest.md#jobs)).

A bare image carries no record. A file with the instrument's record in it is
saved by the API that took the frame -- a still, a run, a recording -- and
never by your program from the array ([Saved files](#saved-files)).

## Saved files

**What a still holds.** A still is saved under the live folder in the format
the settings name (`OutputFormat`: TIFF, OME-TIFF or JPG; a run may instead
build an OME-TIFF hyperstack per well). A TIFF or OME-TIFF keeps mono pixels.
With false color off they are gray and carry no color. With it on, the
channel's color (gray for BF, PC and DF, so they stay gray) is carried beside
them for display: an 8-bit file as a palette, a full-depth one as an ImageJ
lookup table or OME `Channel.Color`. The exception is `FALSE_COLOR_RGB`, which
bakes it into three-channel pixels. A JPG is the image as it is shown: 8-bit,
gray or its channel's color baked into its pixels, and no record in the file.
A manual still's record is the `SavedStill` its capture answered.

**The image mode decides the depth and the encoding.** `ImageMode.EIGHT_BIT`
captures and saves 8-bit mono, and saves a sum of frames as it is shown,
saturating at one frame's white. The full-depth modes capture what the camera
has, 12 bits where it delivers 12. `SCIENTIFIC` saves it right-aligned: a
12-bit frame is 0 to 4095 in a 16-bit container. `SCALED` shifts it to fill the
container. `FALSE_COLOR_RGB`, with false color on, saves it as three-channel
RGB in the channel's color. The file states the depth its values
reach, its record's `significant_bits` -- 12 for a 12-bit `SCIENTIFIC` frame,
16 for a `SCALED` one -- so divide by `(1 << significant_bits) - 1` to scale
it. Every TIFF also carries it in private tag 65123, and an OME-TIFF in its
schema's `SignificantBits`, so a reader that knows neither LumaViewPro's record
nor OME still finds it. A still's or a run's image also carries private tag
65124: the depth one frame's white is at. For a single frame it is the same as
65123 (8 and 8 for an 8-bit still). For a full-depth sum it is lower: the sum
states the bits it can reach, 14 for four 12-bit frames, and 65124 is 12. Every sum states
`frames_summed` (`FramesSummed`), how many it holds.

**The record in the file.** Each TIFF and OME-TIFF carries what the instrument
was when the frame was taken. It records the instrument (model, serial,
camera), the objective and pixel size, the plate and well, and the position (a
recorded frame also whether the stage was moving). It records each lit
channel's current and when it was read, the excitation, the exposure, gain,
black level and binning, and whether the frame was dark or saturated. It also
records the timestamps and where they came from, whether the camera held the
channel's settings, and the frame-validity judgments. An OME-TIFF keeps the
exposure and position in its OME `Plane` and the rest in a `MapAnnotation`
(namespace `etaluma.com/lvp`). A plain TIFF carries the same keys as JSON, its
`Plane` block, in its second `ImageDescription`. Those are the typed record
LumaViewPro reads back. Every file also
carries the record as a person reads it, plain `key = value` pairs (a plain
TIFF's first `ImageDescription`, an ImageJ TIFF's `Info`), which an image
viewer lists with the file's metadata (in Fiji, the Bio-Formats importer's
*Display metadata*). Its keys are the API's words where the API has one fact
for them (`exposure_ms`, `z_um`), and differ where the file states a fact its
own way: `camera_held_channel` where `FrameRecord` has `held_channel`,
`plate_pos_mm.x` and `.y` the pair a `SavedStill` states as `x_mm` and `y_mm`,
and `illumination_ma` the one current, with
`illumination_by_channel_ma.<channel>` each lit channel's, where the record
keys its currents by LED address. The channel and its light are `channel`,
`camera_held_channel`, `illumination_ma` and `dark`. `filterset` is the scope's
filter set, whatever channel was imaged. Every key, its unit, what a sum
does to it, its name in the Plane block and every key an older file states
it under is the reference's [saved-file
facts](../api-reference.md#saved-file-facts). The Plane block keeps its own
names, OME's and LumaViewPro's, for the readers that parse it, so a view key
and its Plane name can differ in one file (`camera_timestamp_tick_hz` and
`TimestampCameraTickHz`). A fact the frame did not know is
absent, never a stand-in. A few values name a type of the reference.
`illumination_timing` is a [`LightTiming`](../api-reference.md#LightTiming),
when the currents were read. Each `judgments.<n>` (the Plane's `FrameValidity`) is a
[`Judgment`](../api-reference.md#Judgment) of the frame against one change:
`counts`, `frames_after`, `after_write_ms`, `after_in_force_ms`. It comes with
its [`Change`](../api-reference.md#Change)'s `source` and `skip`, the frames it
needed, and `counted_by`, each counted frame's number and the
[`Counter`](../api-reference.md#Counter) that counted it (`287:drain`).

The API has no reader of a saved file's record: read it with any TIFF reader,
or keep the `SavedStill` or the run's record the capture answered. In Python:

```python
import tifffile

session.show_channel('BF')
session.light_shown_channel(True)
path = session.manual_capture.start_save_still().result(timeout=30)[0].file
session.light_shown_channel(False)

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
an OME-TIFF per well and scan, its channels colored by OME `Channel.Color`. In
Fiji, open it with `Plugins > Bio-Formats > Importer`, color mode Composite.

**Recordings.** A recording is saved as numbered TIFFs, each frame with its
own record (position, whether the stage was moving, the channel and its
current), or as an MP4 played at the frames' own times. The numbered TIFFs
share their folder with a `recording_manifest.json`. An MP4's manifest is named
for the video: a manual recording writes `Manual/Video_<date>_<time>.mp4` and
`Manual/Video_<date>_<time>_manifest.json` in the live folder. Either manifest
holds what was measured: the frames delivered, their
timestamps and the rate. A rate is a limit: with one set, the manifest's
`frame_budget` is the frames the limit and the duration allow, and a camera
slower than the limit ends at the duration with fewer (`frames_selected`).
No frame is dropped for a slow disk, so saving can run on after the recording
ends. A frame lost to a write error, or one that arrived without the camera's
timestamp, is counted and reported. How long saving ran on is the manifest's
`save_wait_s`, a manual recording's answer (`RecordingFinished.save_wait_s`)
and a run's (`RunOutcome.video_save_wait_s`). A wait over 10 s is reported once
saving finishes. A recording's frames are built into a hyperstack of their
channel when `video_recording_format` is hyperstack, whatever the run format.
The run format stacks only stills.

**A manifest's former words.** A manifest an older build wrote may carry a
word today's does not. Its `manifest_version` does not tell them apart, so a
reader of older manifests reads each word below as the one beside it. Each
row is the word, the key it was a value of (or *key* for a key), the builds
that wrote it and the commit that changed it, and the word that replaced it.

| Word | Of | Written | Now |
|---|---|---|---|
| `camera` | `timestamp_grade` | through 4.0.0-beta38, to `b7d925a45` | `camera_frame_start` |
| `host` | `timestamp_grade` | through 4.0.0-beta38, to `b7d925a45` | `host_arrival` |
| `user_stop` | `end_reason` | 4.0.0-beta29 through beta47, to `a1f4a6793` | `stopped` |
| `disk_floor` | `end_reason` | 4.0.0-beta29 through beta47, to `33c0f3b77` | `disk_space_critical` |
| `short_delivery` | *key* | 4.0.0-beta29 and beta30, to `d869046bc` | removed |
