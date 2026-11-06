# Python

What a Python caller in the same process uses and a REST client does not: bringing a session up, logging, hearing bring-up and outcomes, a still as an image, live frames, operations, and listeners.

## Bringing a session up in Python

**Settings.** A LumaViewPro installation keeps the user's configuration in
`data/current.json`, the live file, and ships `data/settings.json`, the template
a new key's value comes from. `ScopeSession.load_user_settings` reads the user's
configuration as LumaViewPro does: the live file, checked against the template,
repaired, and completed with keys newer releases added. Settings are an argument
of `create`, never read behind your back.

```python
from lumascope import ScopeSession

root = '.'                                   # the LumaViewPro source folder, which holds data/
settings = ScopeSession.load_user_settings(root)
session = ScopeSession.create(
    settings,
    simulate=True,                           # False (the default) opens the real hardware
    settings_file=ScopeSession.user_settings_file(root),   # where accepted writes are saved
)
session.run_application_session()           # the startup motion: home, then the turret
scope = session.scope
```

**What `create` reads from the settings.** A dict you build yourself is
completed from the template, and each of its values is judged as a write of it
would be: a value outside its range is refused (`SettingRefusedError`, naming
the path), never clamped. When the motor board reports a model the catalogue
knows and it differs from `settings['microscope']`, bring-up writes the board's
model into the dict you passed: the hardware outranks the stored selection.
`settings['simulator_tier']` picks the simulator's fidelity
([The simulator](simulator.md#the-simulator)).

**Saving.** Given `settings_file`, the session saves every accepted write
itself, within a second and at shutdown. Without it nothing is saved, which is
what a test or a one-off script usually wants.

**Logging.** Importing the API configures no logging: your `basicConfig()`,
the root logger and your own records stay yours. The session's first call
(`load_user_settings` or `create`) starts LumaViewPro's log folder,
`logs/LVP_Log` under the data root, so a script leaves the support bundle
without a line of setup. The files hang off LumaViewPro's own loggers (`LVP`
and `lvp_logger`), and their main-log records also reach your logging as any
library's do; `logging.getLogger('LVP').setLevel(logging.WARNING)`, and the
same for the `lvp_logger` logger, quiets them to warnings. The trace lanes (serial,
camera, protocol, ...) stay in their own files. Call
`lumascope.start_logging(log_files=False)` before the first session call to
open no file at all, or `lumascope.start_logging()` to start the folder before
it. Logging starts once per process: a start asking for another logging than
the process already has is refused (`LoggingStartRefusedError`,
`'logging_started_otherwise'`), and changes nothing.

**Hearing bring-up.** Bring-up has happened by the time `create` returns. What
it found -- a camera not found, a stored value replaced, a part missing -- is
reported as outcomes; to hear them, pass your listener to `create`
(`outcome_listener=`). A client that comes later reads the record instead:

```python
record = session.bring_up_record()
for part in record.parts:
    print(part.part, 'up' if part.up else 'not up')
assert record.missing == ()                # the simulator has every part
```

**Hearing it come up.** Bring-up takes a few seconds on hardware. To say what
it is doing while it runs -- a progress bar, a status line -- pass
`on_progress=`, the same callback the support report takes: it is called
before each step, on the thread that called `create`, with the share of the
steps already done and the step's words (`Connecting the Motor Controller`,
`Connecting the LED Board`, `Connecting the Camera`, `Configuring the scope`).
LumaViewPro calls `create` on a thread of its own and shows these steps while
its window stays live.

**Shutting down.** `session.shutdown()` finishes what the session is doing --
it stops a live run and recording, then waits for their files and any
post-processing to be written -- and then turns the LEDs off, stops motion and
disconnects the scope. From its first step, new work is refused
(`'session_closing'`). A second call does nothing.

```python
session.shutdown()
assert session.live_work.closed
```

## Hearing outcomes

An outcome nobody asked for is delivered to every outcome listener in the
process, as a [`Notification`](../api-reference.md#Notification). `shown` is
whether it is for display now: an outcome muted when it happened -- during an
unattended run, inside the duplicate window, at shutdown -- arrives with `shown`
False, and again with the same `outcome_id` and `shown` True if it is later
shown. A program that displays opens only the shown ones; one that records keeps
them all. Each carries the outcome's `reason` and `cause`, so a listener
branches on them as a caller does on a raised refusal; a post with no outcome
behind it has no `cause`.

```python
heard = []

def on_outcome(note):
    heard.append(note)
    if note.shown:
        print(note.kind, note.title, '--', note.message)

session.add_outcome_listener(on_outcome)
```

Over REST every outcome is an `outcome` event on the event stream
([REST](rest.md)).

## A still

`session.capture_still(channel)` takes one frame and returns a
`CapturedStill`: the read-only `image` and its `record`, a `FrameRecord` of
what the instrument was when the frame was taken. `channel` is the shown
channel (another is refused `'channel_not_shown'`), and another LED lit is
refused `'other_led_lit'`; `capture_still()` with no channel takes the frame
with the camera and the LEDs as they are.

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

## Live frames

`scope.imaging.add_frame_listener(callback)` hands each frame
the camera delivers to `callback(frame)`, a read-only `Frame` (`image`, `seq`,
`timestamp`, `arrived_ns`, `significant_bits`, `chunks`), on the camera's own
thread. A listener has 24 ms per frame: one that overruns its budget or raises
30 frames in a row is removed, and an outcome says so. Copy what you keep, and
return quickly.

```python
import threading

arrived = threading.Event()
frames = []

def on_frame(frame):
    frames.append(frame.seq)
    arrived.set()

scope.imaging.add_frame_listener(on_frame, name='guide')
assert arrived.wait(timeout=10)
scope.imaging.remove_frame_listener(on_frame)
```

## Operations

**An Operation** is the handle every `start_` member returns, typed by what its
`result()` answers: `Operation[RunOutcome]` for a run, `Operation[None]` for a
move or a home, `Operation[tuple[SavedStill, ...]]` for a still and
`Operation[str]`, its end reason, for a recording. `result()` waits
for the run to let go of the scope and answers its outcome; `done()` says
whether it has; `progress()` answers where it has got; `stop()` asks it to stop
and returns at once -- the run unwinds, puts the scope back, and ends
`aborted`. A run's images keep writing after it lets go: `files` is their own
operation. In-process each run also has a waiting form, `run_single_scan`,
`run_protocol`, ...; over REST a run is a job ([REST](rest.md)).

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

## Events and listeners

Each change the scope makes is delivered to the listeners registered for it,
and over REST as an event on the event stream. The
[events table](../api-reference.md#events) lists every event, the member that
delivers it, its record (the callback's arguments, in order) and how it
reaches a REST client. Each `add_*_listener` has its `remove_*_listener`.

```python
import threading

settled = threading.Event()

def on_position(axis, at):
    if axis == 'Z' and at.state == 'idle':
        settled.set()

scope.motion.add_position_listener(on_position)
scope.motion.move_z(4500.0)
assert settled.wait(timeout=10)
scope.motion.remove_position_listener(on_position)
```

**Threads.** A listener is called on the thread that made the change -- the
IO lane for a move or an LED, the camera's thread for a frame -- so it returns
quickly and does not call back into the API and wait. A GUI host that needs its
callbacks on its UI thread tells the process once, before its first session,
through `ScopeSession.set_ui_dispatcher`; a script never sets it.

**A run's events** are the `RunEvents` handlers given to its start:
`scan_started`, `step_started`, `frame_captured`, `scan_ended`, `run_ended`
and `files_written`, in that order. `run_ended` comes once the run has let go
of the scope, and `files_written` once its images are on disk: chain the next
work from `files_written`, not from `run_ended`.

**Levels and edges.** The run-state listener is a level: it is called once at
registration with the state now, then on every change, so a display that
registers late is not wrong. Every other listener hears changes from its
registration on; read the current value first.
