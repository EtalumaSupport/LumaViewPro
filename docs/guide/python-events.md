# Python: live frames and listeners

What a Python caller hears as it happens: the camera's live frames, and the events every change sends.

## Live frames

`scope.imaging.add_frame_listener(callback)` hands each frame
the camera delivers to `callback(frame)`, a read-only `Frame` (`image`, `seq`,
`timestamp`, `arrived_ns`, `significant_bits`, `chunks`), on the camera's own
thread. A listener has 24 ms per frame: one that overruns its budget or raises
30 frames in a row is removed, and an outcome says so. Keeping a frame is safe:
it is read-only and no later frame changes it, so draw on a copy. Return
quickly.

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

## Events and listeners

Each change the scope makes is delivered to the listeners registered for it,
and over REST as an event on the event stream -- but the frame events, which
the live view carries. The
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
quickly and does not call back into the API and wait. A listener's raise is
reported once and the other listeners are still told, so a listener need not
guard itself. A GUI host tells the process once, before its first session,
how to reach its UI thread (`ScopeSession.set_ui_dispatcher`): run events given
`on_ui_thread=True` and the lanes' completions arrive there, while a listener
still schedules its own drawing. A script never sets it.

**A run's events** are the `RunEvents` handlers given to its start:
`scan_started`, `step_started`, `frame_captured`, `scan_ended`, `run_ended`
and `files_written`, in that order, and `video_progress` as a video step
records and writes. `run_ended` comes once the run has let go of the scope, and
`files_written` once its images are on disk: chain the next work from
`files_written`, not from `run_ended`. A composite's `run_ended` comes once its
merge settles, which may be after its `files_written`. A handler's raise is
reported under its event's name.

**Levels and edges.** The run-state listener is a level: it is called once at
registration with the state now, then on every change, so a display that
registers late is not wrong. Every other listener hears changes from its
registration on; read the current value first.
