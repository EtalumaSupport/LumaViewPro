# Protocols and runs

Protocols, runs, and post-processing.

**A protocol** is a list of steps -- each a place on the protocol's plate, a
channel, and how to capture it -- with a schedule. Three members give you one:
`session.new_protocol(...)` builds one over every well of the selected plate,
`session.create_empty_protocol()` starts one with no steps, and
`session.load_protocol(path)` reads a file and puts the scope on its plate -- a
write, refused under another's control session and when the scope cannot be on
that plate. `new_protocol` and `add_step` capture each channel whose acquire
setting is on (`set_channel_acquire`); `add_step(channels=...)` names the
channels instead, one acquiring nothing added as an image step. A step captures
an image or a video, as its channel's acquire mode says
([`AcquireMode`](../api-reference.md#AcquireMode)), and carries each
fluorescence channel's stimulation (`StepRecord.stim_config`, a
[`Stimulation`](../api-reference.md#Stimulation) per channel).

```python
session.set_channel_acquire('BF', 'image')       # BF is captured by every step
session.update_settings('BF.illumination_ma', 5)

protocol = session.create_empty_protocol()
scope.motion.move_to_well('B2')
print(session.add_step(protocol, channels=('BF',)))     # a step here, on the protocol's plate
print(protocol.step_records()[0])

path = session.save_protocol(protocol, session.live_folder_path('Protocols/guide.tsv'))
assert session.load_protocol(path).num_steps() == 1
```

```python
session.update_settings('zstack.range', 20.0)    # a stack's extent and step are settings
session.update_settings('zstack.step_size', 5.0)
plate = session.new_protocol(tiling='2x2', use_zstacking=True)
print(plate.num_steps(), plate.tiling())
```

A z-stack whose channel autofocuses is focused once, by a sweep fired at its
first slice and centred on its reference slice, and the found focus lands on
that reference slice -- the slice built at the channel's focus: the middle
slice of a centre stack (the lower middle of an even count), the top of a
top stack, the bottom of a bottom stack. Every slice keeps its distance from
that slice, its record's `z_stack_offset_um`, whichever of its slices a
caller turned autofocus on or moved.

`update_step` on a slice of a z-stack writes the channel's light and camera
to every slice of that channel in the stack, and the live X, Y and objective
to every slice of the stack, every channel. The slice alone takes the live Z,
its label changes only when `label` is given, and naming another channel is
refused (`'step_value_refused'`). So a caller that updates a slice without
first going to it (`go_to_step`) moves the whole stack to where the stage is.

A file in which one z-stack spans several tiles is read as one stack per tile,
with a warning in the log.

A protocol is edited only through the session (`add_step`, `update_step`,
`rename_step`, `delete_step`, `apply_tiling`, `apply_zstacking`,
`set_protocol_schedule`, `set_protocol_labware`, `set_protocol_capture_root`,
`set_step_autofocus`, `apply_focus_to_channel_steps`,
`point_steps_at_turret_slot`, ...), and each edit is a write, refused under
another's control session. Its records (`step_records()`) are read-only, and a
run works on its own copy. `protocol_size_advisory(protocol)` says, before a
run, whether it is large enough to warn about, and
`start_go_to_step(protocol, step_index)` moves the scope to a step: its
objective, its X, Y and Z.

**The schedule** is in seconds in the API: `period_s` between scans and
`duration_s` in all. The protocol file keeps them in minutes and hours.

```python
session.set_protocol_schedule(protocol, period_s=600.0, duration_s=3600.0)
assert protocol.num_scans() == 6
```

**The protocol file** is tab-separated. Its first row is
`Protocol.PROTOCOL_FILE_HEADER`, then the version, the period (minutes), the
duration (hours), the plate and the capture root; then a Layer Settings block,
a row of each channel's settings; then a table of steps whose columns are
`Protocol.CURRENT_COLUMNS`. `save_protocol` writes the block from each channel
acquiring in the settings; the copy a run saves in its folder has it only when
its protocol was read from a file that had one, so a protocol built in memory
runs to a file with none:

```python
from lumascope import Protocol

print(Protocol.CURRENT_VERSION, Protocol.CURRENT_COLUMNS)
```

`X` and `Y` are plate millimetres from the plate's top-left, `Z` stage
micrometres, `Illumination` milliamps, `Gain` dB, `Exposure` milliseconds, and
`Sum` the frames summed. A file of version 2 or later is converted when it is
read; version 1 and a newer one than this release's are refused
(`ProtocolFormatError`). A load refuses a plate the installation does not know
(`ProtocolFormatError`); glass this scope cannot put in the light path, or a
step on a channel it does not have, with the reason a run would give
(`ProtocolRunRefusedError`; `'channel_unknown'` for a channel no release has);
a Layer Settings value no channel setting takes (`'values_out_of_range'`); and a
schedule no run could keep (`ProtocolScheduleRefusedError`). A step's value is the one thing
loading is lenient about: one no run could use is reported when the file is
read, and refused when a run starts.

**Runs.** A run is started by the session's protocol runner and returns an
`Operation` at once:

| Member | Runs |
|---|---|
| `start_single_scan(protocol)` | every step once |
| `start_protocol(protocol)` | every scan of the schedule |
| `start_autofocus(channel)` | an autofocus where the stage is |
| `start_autofocus_all_steps(protocol)` | an autofocus at each step (once per z-stack), written into the protocol when the scan completes and its steps did not change (`RunOutcome.focus_written`) |
| `start_zstack(channel)` | a stack of the stored range where the stage is |
| `start_composite(...)` | one capture per acquiring channel, merged |

```python
from lumascope import RunStatus

runner = session.protocol_runner()
started = runner.start_single_scan(protocol, sequence_name='guide')
outcome = started.result(timeout=120)
assert outcome.status is RunStatus.COMPLETED, outcome.message
print(outcome.captures)                     # asked, captured, failed, saturated, dark
print(started.files.result(timeout=60))     # the run's images: written, or how many were not
```

A run that cannot start is refused before it takes the scope, mostly
`ProtocolRunRefusedError`, its `reason` naming why (`'position_unknown'`,
`'zstack_not_configured'`, `'exclusive_activity_running'`, ...); each start's
`Raises:` lists the rest. Once started, a run
always ends with a `RunOutcome` -- it does not raise -- whose `status` is one of
`completed`, `incomplete`, `aborted`, `failed` or `failed_at_start`, with a
`reason` and the words a person reads.

**A run's files.** A run lets go of the scope before its images are on disk:
its operation ends with the run, and `files`, its own operation, once they are
written ([Operations](python.md#operations); over REST, the run's job and its
`files` job, [Jobs](rest.md#jobs)). Read the run's folder, or start the next
run, after `files`. Its result, a
[`RunFiles`](../api-reference.md#RunFiles), counts the run's images:
`outcome` is `written` when every image captured is on disk, else
`incomplete`, with `not_written` and `not_written_reason`; `written` is the
images on disk, so a folder of two images and its protocol and record says
`written` 2. In a time-lapse each autofocus step starts its next
interval's sweep from the focus it last found, kept in the run's copy; the
protocol you passed keeps its Z.

**Where a run writes.** Under the live folder, unless its `parent_dir` names
another:

| Run | Folder |
|---|---|
| a scan or a protocol | `ProtocolData/<date>_<time>`: an image per step and channel, the protocol it ran, a record of each capture |
| a z-stack | `Manual/Z-Stacks` |
| a composite | `Manual/Composites` |
| an autofocus | nothing, unless `save_characterization_data` asks: `Autofocus Characterization` |

A run's video steps record as a manual recording does
([Saved files](captures-and-files.md#saved-files)), and its hyperstacks are built once the run ends.

**A scan's folder** holds, for each step and channel, its image
`<step name>_<scan>.<ext>` (`custom0000_BF_0000.tiff`: step `custom0000_BF`,
scan 0); the protocol it ran, `<sequence_name>.tsv` (named by the start's
`sequence_name`; given none, a single scan's is scan.tsv and a protocol's
protocol.tsv); and
`protocol_record.tsv`, one row per capture after a short header naming that
protocol file. Its columns: `Filename`, `Step Name`, `Step Index` (the step's
row, from 0), `Scan Count` (the scan, from 0), `Timestamp` (with its UTC
offset), `Frame Count` (1 for a still, the frames written for a video step,
0 for a capture that left no file) and `Duration (s)` (a video step's). A
capture taken with saving off is named `unsaved`.

**Where a run stands** is its operation's `progress`, a
[`RunProgress`](../api-reference.md#RunProgress) -- `is_live`, `step_index`,
`scans_remaining`, `run_dir` and the rest; over REST it is the run's job's
`progress`, as a poll of the job reads it. Its `period_s` is the protocol's
schedule, which a single scan reports and does not keep: an empty protocol
takes the settings' (`protocol.period`, shipped 20 minutes, `period_s` 1200).

**Post-processing** builds derived files from a run's folder:
`session.post_processing.build(folder, zproject=..., stitch=..., composite=...,
video=...)` builds the projections first, then the stitches, the composites
and the videos, each group of a run from every frame it recorded or not at all.
Each build takes only what the one before it in the call carried forward, so
`zproject` and `stitch` together stitch the projections, and the same call made
again rebuilds nothing already made and resumes where it stopped. A build does
not start while a run, a recording, a home or a diagnostic holds the scope
(`'scope_held'`) or a run's files are being written (`'files_writing'`); one
already running when a run starts runs on, on its own lane.

`session.post_processing` also enhances an image (`enhance`), counts cells
(`count_cells`, with saved methods: `save_cell_count_method`,
`load_cell_count_method`) and draws graphs of what was counted (`graph`).
