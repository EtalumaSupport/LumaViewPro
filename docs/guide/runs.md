# Protocols and runs

Protocols, runs, and post-processing.

**A protocol** is a list of steps -- each a place on the protocol's plate, a
channel, and how to capture it -- with a schedule. Three members give you one:
`session.new_protocol(...)` builds one over every well of the selected plate,
`session.create_empty_protocol()` starts one with no steps, and
`session.load_protocol(path)` reads a file. Which channels a built protocol
captures is each channel's acquire setting (`set_channel_acquire`).

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
to every slice of the stack, every channel; the slice keeps its own Z and
label. So a caller that updates a slice without first going to it
(`go_to_step`) moves the whole stack to where the stage is.

A protocol read to run gives each tile of a z-stack its own group, so a file
in which one group spans several tiles (saved by a build that tiled an
already-stacked protocol) reads as one stack per tile, with a warning in the
log.

A protocol is edited only through the session (`add_step`, `update_step`,
`rename_step`, `delete_step`, `apply_tiling`, `apply_zstacking`,
`set_protocol_schedule`, ...), and each edit is a guarded write, refused under
another's control session. Its records (`step_records()`) are read-only, and a
run works on its own copy.

**The schedule** is in seconds in the API: `period_s` between scans and
`duration_s` in all. The protocol file keeps them in minutes and hours.

```python
session.set_protocol_schedule(protocol, period_s=600.0, duration_s=3600.0)
assert protocol.num_scans() == 6
```

**The protocol file** is tab-separated. Its first row is
`Protocol.PROTOCOL_FILE_HEADER`, then the version, the period (minutes), the
duration (hours), the plate and the capture root, then a table of steps whose
columns are `Protocol.CURRENT_COLUMNS`:

```python
from lumascope import Protocol

print(Protocol.CURRENT_VERSION, Protocol.CURRENT_COLUMNS)
```

`X` and `Y` are plate millimetres from the plate's top-left, `Z` stage
micrometres, `Illumination` milliamps, `Gain` dB, `Exposure` milliseconds, and
`Sum` the frames summed. A file of an earlier version is converted when it is
read; a newer one is refused (`ProtocolFormatError`). Loading is strict about a
cell's type and lenient about its value: a value no run could use is reported
when the file is read, and refused when a run starts.

**Runs.** A run is started by the session's protocol runner and returns an
`Operation` at once:

| Member | Runs |
|---|---|
| `start_single_scan(protocol)` | every step once |
| `start_protocol(protocol)` | every scan of the schedule |
| `start_autofocus(channel)` | an autofocus where the stage is |
| `start_autofocus_all_steps(protocol)` | an autofocus at each step (once per z-stack), written into the protocol |
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

A run that cannot start is refused before it takes the scope:
`ProtocolRunRefusedError`, its `reason` naming why (`'position_unknown'`,
`'zstack_not_configured'`, `'exclusive_activity_running'`, ...). Once started, a run
always ends with a `RunOutcome` -- it does not raise -- whose `status` is one of
`completed`, `incomplete`, `aborted`, `failed` or `failed_at_start`, with a
`reason` and the words a person reads.

**Where a run writes.** Under the live folder, `ProtocolData/<time>`: one
image per step and channel, the protocol it ran, and a record of each capture.
A run's video steps record as a manual recording does
([Saved files](captures-and-files.md#saved-files)), and its hyperstacks are built once the run ends.

**Post-processing** builds derived files from a run's folder:
`session.post_processing.build(folder, zproject=..., stitch=..., composite=...,
video=...)` builds the projections first, then the stitches, the composites
and the videos, each group of a run from every frame it recorded or not at all.
A build runs on its own lane, beside the next run, and is refused while the
run it would read is still writing.
