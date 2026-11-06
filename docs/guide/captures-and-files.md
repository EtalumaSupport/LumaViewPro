# Captures and files

Frame validity, which channel a still is, saved stills, and what a saved file holds.

## Frames and captures

**Frame validity.** After a change -- an LED lit, an exposure set, the stage
moved -- the camera keeps delivering frames that were exposed before the change
took effect. A capture waits until a frame counts for every recorded change,
by counting frames: a frame counts once the change's source's count of frames
has arrived since the change (a move's since its stop), the count the camera's
timing record measured, else the shipped one. `scope.capabilities.frame_rules`
answers each source's count, and `scope.imaging.changes_in_force()` whether a
frame arriving now would count
(`GET /api/v1/scope/imaging/changes_in_force`). A capture never infers validity from pixel content; when the scope keeps
changing longer than its budget, it raises `CaptureFailedError`
(`'deadline_expired'`). A recording, manual or a run's video step, waits the
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
LumaViewPro's Capture button takes, and saves it: the folder, the name, the
summing and the format come from the settings. It returns an `Operation` of a
`SavedStill` per file written, each its `file` and the record of it.

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

A bare image carries no record. A file with the instrument's record in it is
saved by the API that took the frame -- a still, a run, a recording -- and
never by your program from the array ([Saved files](#saved-files)).

## Saved files

**What a still holds.** A still is saved under the live folder in the format
the settings name (`OutputFormat`: TIFF, OME-TIFF or JPG; a run may instead
build an OME-TIFF hyperstack per well). A fluorescence
still is mono: its false colour is a choice made when it is shown, never in its
pixels. A frame deeper than 8 bits keeps the sensor's values right-aligned (a
12-bit frame is 0 to 4095 in a 16-bit container), and the file states the true
depth as `SignificantBits`: divide by `(1 << SignificantBits) - 1` to scale it,
never by 65535. A sum of frames states the bits it can reach -- 14 for four
12-bit frames -- and `FramesSummed` how many it holds.

**The image mode decides the depth and the encoding.** `ImageMode.EIGHT_BIT`
captures and saves 8-bit mono. The full-depth modes capture what the camera
has -- 12 bits where it delivers 12 -- and save it right-aligned
(`SCIENTIFIC`), shifted to fill the 16-bit container (`SCALED`), or as
three-channel RGB in the channel's colour (`FALSE_COLOR_RGB`); each states its
`SignificantBits`.

**The record in the file.** Each file carries what the instrument was when the
frame was taken: the objective, the position, each lit channel's current and
when it was read, the exposure, gain, black level and binning, the timestamps
and where they came from, whether the camera held the channel's settings, and
the frame-validity judgments. An OME-TIFF keeps the exposure and position in
its OME `Plane` and the rest in a `MapAnnotation` (namespace `etaluma.com/lvp`);
a plain TIFF carries the same keys in its `Plane` block. Those are the typed
record LVP reads back. Every file also carries the record as a person reads it,
plain `key = value` pairs under the record's own names (`plate.well_label = D6`,
`exposure_time_ms = 50.0`): Fiji lists them in the Bio-Formats importer's
*Display metadata* for every file, and in *Image > Show Info* for a plain TIFF.
Every key, its unit and what a sum does to it is the reference's [saved-file
facts](../api-reference.md#saved-file-facts). A fact the frame did not know is
absent, never a stand-in.

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
Every frame is kept however slowly the disk saves them, so saving can run on
after the recording ends: the manifest's `save_wait_s`, a manual recording's
answer (`RecordingFinished.save_wait_s`) and a run's
(`RunOutcome.video_save_wait_s`) say how long, and a wait over 10 s is told
once saving finishes. Frames of one channel are also built into a hyperstack.
