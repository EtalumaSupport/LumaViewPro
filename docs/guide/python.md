# Python

What a Python caller in the same process uses and a REST client does not:
bringing a session up, logging, hearing bring-up and outcomes, a still as an
image, and operations. Live frames and listeners are
[python-events.md](python-events.md).

## Bringing a session up in Python

**Settings.** A LumaViewPro installation keeps the user's configuration in
`data/current.json`, the live file, and ships `data/settings.json`, the template
a new key's value comes from. `ScopeSession.load_user_settings` reads the user's
configuration as LumaViewPro does: the live file, checked against the template,
repaired, and completed with keys newer releases added. Where LumaViewPro would
fall back to the template and ask, it refuses: a `current.json` it cannot use is
`SettingsFileError`, and a folder that is no installation `'not_an_installation'`.
Called inside a running LumaViewPro, by a plugin, it answers the settings
LumaViewPro already prepared. Settings are an argument of `create`, never read
behind your back.

```python
from lumascope import ScopeSession

root = '.'                                   # the LumaViewPro source folder, which holds data/
settings = ScopeSession.load_user_settings(root)
session = ScopeSession.create(
    settings,
    source_path=root,                        # the installation the scope is built from
    simulate=True,                           # False (the default) opens the real hardware
    settings_file=ScopeSession.user_settings_file(root),   # where accepted writes are saved
)
session.run_application_session()           # the startup motion: home, then the turret
scope = session.scope
```

`create` warns once (`FutureWarning`) that the API is pre-release until
`4.0.0` ([the note](../lumascope-skills.md)). Its other keywords are
[its reference's](../api-reference.md#ScopeSession.create).
`run_application_session(disable_homing=True)` runs no startup motion at all.

**What `create` reads from the settings.** A dict you build yourself is
completed from the template, and each of its values is judged as a write of it
would be: a value outside its range is refused (`SettingRefusedError`, naming
the path), never clamped. When the motor board reports a model the catalogue
knows and it differs from `settings['microscope']`, bring-up writes the board's
model into the dict you passed: the hardware outranks the stored selection.
`settings['simulator_tier']` picks the simulator's fidelity
([The simulator](simulator.md#the-simulator)).

**Saving.** Given `settings_file`, the session saves every accepted write
itself, within a second and at shutdown, once the scope is up. While the
settings are provisional (`session.settings_are_provisional`), or when the
scope found no hardware at bring-up, writes are kept unsaved and the session
says so once (`SettingsNotSavedNotice`). Without `settings_file` nothing is
saved, which is what a test or a one-off script usually wants.

**Logging.** Importing the API configures no logging: your `basicConfig()`,
the root logger and your own records stay yours. The session's first call
(`load_user_settings` or `create`) starts LumaViewPro's log folder,
`logs/LVP_Log` under the data root, so a script leaves the support bundle
without a line of setup. The files hang off LumaViewPro's own loggers (`LVP`
and `lvp_logger`), and their main-log records also reach your logging as any
library's do. Their level is the setting `logging.default.level` (shipped
`VERBOSE`), which `create` applies and the end of the startup motion applies
again, so a `setLevel` on those loggers before then is replaced: to quiet them,
set `logging.default.level` (a level's name, such as `WARNING`) in the settings
handed to `create`. The trace lanes (serial, camera, protocol, ...) stay in
their own files. Call
`lumascope.start_logging(log_files=False)` before the first session call to
open no file at all, or `lumascope.start_logging()` to start the folder before
it. Logging starts once per process: a start asking for another logging than
the process already has is refused (`LoggingStartRefusedError`,
`'logging_started_otherwise'`), and changes nothing.

**Hearing bring-up.** Bring-up has happened by the time `create` returns. What
it found -- a camera not found, a stored value replaced, a part missing -- is
reported as outcomes; to hear them, pass your listener to `create`
(`outcome_listener=`). A client that comes later reads the record instead, and
`scope.no_hardware` (no part came up) and `scope.motion_expected` (False for a
model with no motion); the record's `substitutions` and `settings_set_aside`
name what bring-up replaced:

```python
record = session.bring_up_record()
for part in record.parts:
    print(part.part, 'up' if part.up else 'not up')
assert record.missing == ()                # the simulator has every part
```

**Hearing it come up.** Bring-up takes a few seconds on hardware. To say what
it is doing while it runs -- a progress bar, a status line -- pass
`on_progress=`, the same callback the support report takes: it is called
before each step, on the thread that called `create`, with the percentage of
the steps already done (0, 25, 50, 75) and the step's words (`Connecting the Motor Controller`,
`Connecting the LED Board`, `Connecting the Camera`, `Configuring the scope`).
LumaViewPro calls `create` on a thread of its own and shows these steps while
its window stays live.

**Shutting down.** `session.shutdown()` finishes what the session is doing --
it stops a live run and recording, then waits for their files and any
post-processing to be written -- and then turns the LEDs off, stops motion and
disconnects the scope. From its first step, new work that would take the scope
-- a run, a recording, a home, a diagnostic -- is refused `SessionClosingError`
(`'session_closing'`), and once it has closed every command is refused
`'scope_disconnected'`. It waits for the work's files with no time limit;
`discard_close_drain()` drops the video frames still queued, so a close stuck
behind a slow disk can end. A second call during the close waits for it.

```python
session.shutdown()
assert session.live_work.closed
```

## Hearing outcomes

Every outcome the API posts rather than raises is delivered to every outcome
listener in the process, as a [`Notification`](../api-reference.md#Notification):
one nobody asked for, and a notice that answers a call just made (`solicited`
True), such as the model saved for the next start. `shown` is whether it is for
display now: an outcome muted when it happened -- during an unattended run, at
shutdown, or the same title shown less than 10 s before -- arrives with `shown`
False, and again with the same `outcome_id` and `shown` True if it is later
shown. A program that displays opens only the shown ones; one that records keeps
them all. Each carries the outcome's `reason` and `cause`, so a listener
branches on them as a caller does on a raised refusal. A post made with no
outcome behind it has kind `unclassified` and no `cause`; a quiet outcome is
never delivered; `fatal` marks one that reaches a person even while an
unattended run mutes the rest.

```python
heard = []

def on_outcome(note):
    heard.append(note)
    if note.shown:
        print(note.kind, note.title, '--', note.message)

session.add_outcome_listener(on_outcome)
```

Over REST an outcome nobody asked for is an `outcome` event on the event
stream, sent once, at its first delivery: a muted one shown later is not sent
again, and a solicited notice is not sent ([REST](rest.md)).

## A still

In-process: the image does not travel the wire, and over REST a still is
`POST /api/v1/still` ([The live view and files](rest.md#the-live-view-and-files)).
`session.capture_still(channel)` takes one frame and returns a
`CapturedStill`: the read-only `image` and its `record`, a `FrameRecord` of
what the instrument was when the frame was taken. `channel` is the shown
channel (another is refused `'channel_not_shown'`), and another LED lit is
refused `'other_led_lit'`; `capture_still()` with no channel takes the frame
with the camera and the LEDs as they are. The image is 8-bit unless
`force_to_8bit=False` keeps the depth the image mode captures, and `sum_count`
sums frames. The record's `illumination_ma` is keyed by LED board address;
`scope.channel_identity` maps a channel to its addresses.

```python
scope.motion.move_to_well('B2')             # over the simulator's specimen
session.show_channel('BF')
scope.illumination.led_on('BF', 5)
still = session.capture_still('BF', all_ones_check=True)
record = still.record
print(still.image.shape, still.image.dtype)
print(record.held_channel, record.exposure_ms, record.gain_db, record.illumination_ma)
assert record.dark is False                 # lit, and light was seen
assert record.saturated_fraction < 0.98     # measured, because all_ones_check asked
```

## Operations

**An Operation** is the handle every `start_` member returns, typed by what its
`result()` answers: `Operation[RunOutcome]` for a run, `Operation[None]` for a
move or a home, `Operation[tuple[SavedStill, ...]]` for a still and
`Operation[RecordingFinished]` for a recording (its `end_reason`, and how long
saving ran on). `result()` waits
for the run to let go of the scope and answers its outcome; `done()` says
whether it has; `progress()` answers where it has got; `stop()` asks it to stop
and returns at once -- the run unwinds, puts the scope back, and ends
`aborted`. A run's images keep writing after it lets go: `files` is their own
operation. `result()` and `exception()` take a timeout; on a GUI host's UI
thread they are refused (`'wait_on_ui_thread'`) until the operation is done, as
every waiting member is. A still's `stop()` does nothing, and a move's or a
home's stop can be refused or fail as `stop_motion()` can. In-process each run
also has a waiting form, `run_single_scan`, `run_protocol`, ...; over REST a run
is a job ([Jobs](rest.md#jobs)).

```python
import threading
from lumascope import RunEvents, RunStatus

session.set_channel_acquire('BF', 'image')
protocol = session.create_empty_protocol()          # one step, at B2
scope.motion.move_to_well('B2')
session.add_step(protocol, channels=('BF',))
session.set_protocol_schedule(protocol, period_s=600.0, duration_s=3600.0)
runner = session.protocol_runner()

ended = threading.Event()
events = RunEvents(run_ended=lambda outcome, run_dir, protocol: ended.set())
running = runner.start_protocol(protocol, sequence_name='guide_stopped', events=events)
running.stop()
assert ended.wait(timeout=60)
assert running.result().status is RunStatus.ABORTED
```
