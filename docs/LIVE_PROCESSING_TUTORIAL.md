# Live Processing Plugin Tutorial

A `live_processing` plugin runs a small synchronous handler on every camera frame, as soon as that frame arrives from the SDK. Use it for per-frame analysis (histograms, focus scores, motion detection, frame counting) and for sinks that need to react to each frame immediately (manual recording, telemetry).

This tutorial covers what live_processing plugins are, the contract they sign with the imaging pipeline, a minimum viable example, a failure-injection walk-through, and the common pitfalls. The audience is internal devs writing first-party plugins (start here -- intern's Phase C work in particular). External SDK / plugin authors can reach this doc from `LumaViewPro/docs/LumascopeSkills.md` -> "Live frame listeners."

## Status

| Item | State |
|---|---|
| Infrastructure (registry + budget wrapper + drop policy) | DONE 2026-05-19 (LVP `wave7-phase4d5` Phase 4d.5b-e) |
| LumascopeSkills.md "Live frame listeners" entry | DONE 2026-05-19 (paired with this doc) |
| Bench-validated per-handler timing | TBD (4d.5f bench validation; will compare to 24 ms budget under a sample plugin) |
| External-plugin-author audience | Deferred -- internal devs are the practical first audience per WAVE7_PHASE_4D5_PLAN sec 9 #5. |

## When to use live_processing

| You want to... | Use this namespace | Reason |
|---|---|---|
| React to every camera frame (analysis, recording, telemetry) | **`live_processing`** | Handler runs synchronously on the SDK thread, gets every frame |
| Run analysis on saved files at protocol end | `post_processing` | Operates on TIFFs already on disk, not a live stream |
| Add UI widgets to LumaViewPro | `ui` | Mounts a Kivy widget at a known mount point |
| Expose HTTP endpoints | `rest` | (reserved -- ships with REST_API_PLAN.md Phase 1) |

If your work needs the FULL camera frame in memory at >5 fps, this is the namespace. If you can afford to wait until protocol end and operate on disk, prefer `post_processing` -- it's cheaper and unconstrained by the 24 ms budget.

## The contract

When you register a live_processing handler, you sign three contracts:

### 1. Handler signature

```python
def handler(image, timestamp, chunks):
    ...
```

- `image`: `numpy.ndarray` of the frame. Shape and dtype follow the camera (`(H, W)` for Mono8, `(H, W, 3)` for RGB, etc.). Read-only by contract; see "Don't mutate" below.
- `timestamp`: `datetime.datetime`. Host-side arrival time of the frame.
- `chunks`: `dict | None`. Per-frame SDK chunk metadata when supported (Pylon GigE / USB3 chunk mode), `None` otherwise. The simulated camera always passes `None`.

The handler must accept and return nothing useful (return value is ignored).

### 2. Don't mutate the image array

The same `image` array is passed to ALL registered handlers, in registration order, AND is used by downstream display + capture consumers. If your handler does `image[0, 0] = 0`, the next handler sees a corrupted frame and so does the display thread.

If you need to keep results, write to your OWN output buffer:

```python
def handler(image, ts, chunks):
    # GOOD: copy or transform into your own buffer
    self._my_buffer = image.copy()
    self._scores.append(compute_score(image))

def handler(image, ts, chunks):
    # BAD: mutates the shared array
    image -= np.median(image)   # downstream sees noise-subtracted!
```

### 3. The 24 ms budget + drop policy

Each handler invocation is timed. Exceeding the budget logs a WARNING with the elapsed time + the plugin name + the consecutive over-budget count. After 30 consecutive over-budget hits (~1 second at 30 fps) the handler is auto-removed AND a warning notification fires.

Settings live as module constants in `LumaViewPro/modules/lumascope_api/imaging.py`:

```python
HANDLER_BUDGET_MS = 24    # anchored to 30 fps target (FRAME_VALIDITY_RIG_COMPARISON_2026-05-19)
HANDLER_DROP_K    = 30    # consecutive over-budget invocations before auto-remove
```

A handler that **raises an exception** is logged but does NOT count toward the budget counter -- it's a different failure class. The next call gets a fresh shot.

### 4. Re-entrancy is not a concern

The driver's per-frame fire site is single-threaded. You will not be re-entered on the same thread, so you don't need locks or atomic counters to protect handler-local state. (You DO need locks if you stash the array on `self` and read it from another thread later; that's standard producer/consumer territory.)

## Minimum viable plugin

A plugin is a Python package with a `register(ctx)` function discoverable via the entry-point group `lvp.plugins`. The minimum viable live_processing plugin counts frames and logs every 100th:

```python
# my_plugin/__init__.py
import logging
from modules.plugins import PluginSpec

logger = logging.getLogger('lvp_logger')

PLUGIN_SPEC = PluginSpec(
    name='frame_counter',
    version='0.1.0',
    requires_lvp_version='>=4.0.0',
    description='Counts frames and logs every 100th.',
)


_counter = {'n': 0}


def _on_frame(image, ts, chunks):
    _counter['n'] += 1
    if _counter['n'] % 100 == 0:
        logger.info(f'[frame_counter] received {_counter["n"]} frames')


def register(ctx):
    ctx.plugins.live_processing.register(PLUGIN_SPEC, _on_frame)


def unregister(ctx):
    ctx.plugins.live_processing.unregister(PLUGIN_SPEC.name)
```

```toml
# my_plugin/pyproject.toml
[project]
name = "my-frame-counter"
version = "0.1.0"
dependencies = []

[project.entry-points."lvp.plugins"]
frame_counter = "my_plugin"
```

Install with `pip install -e .` from the plugin source. LumaViewPro's `load_plugins(ctx)` discovers it at startup; `unload_plugins(ctx)` calls your `unregister(ctx)` at shutdown.

That's the entire skeleton. The 30 lines above are a working plugin.

## Failure-injection walk-through

This shows what happens when a handler exceeds budget consistently. Save as `slow_plugin/__init__.py`:

```python
import time
from modules.plugins import PluginSpec

PLUGIN_SPEC = PluginSpec(
    name='slow_demo',
    version='0.1.0',
    requires_lvp_version='>=4.0.0',
    description='Demonstrates the budget drop policy.',
)


def _slow_handler(image, ts, chunks):
    # 30 ms per frame -- 6 ms over the 24 ms budget on every call.
    time.sleep(0.030)


def register(ctx):
    ctx.plugins.live_processing.register(PLUGIN_SPEC, _slow_handler)


def unregister(ctx):
    ctx.plugins.live_processing.unregister(PLUGIN_SPEC.name)
```

**What you'll see in `lumaviewpro.log`** at ~30 fps:

```
WARNING [...] live_processing handler 'slow_demo' over budget: 30.4ms (budget 24ms) -- consecutive 1/30
WARNING [...] live_processing handler 'slow_demo' over budget: 30.2ms (budget 24ms) -- consecutive 2/30
...
WARNING [...] live_processing handler 'slow_demo' over budget: 30.1ms (budget 24ms) -- consecutive 30/30
```

After the 30th consecutive over-budget hit, the wrapper auto-removes itself from the listener list AND fires:

```
NOTIFICATION WARNING | Live Processing/Plugin 'slow_demo' removed | The plugin's frame handler exceeded the 24ms budget for 30 consecutive frames (last: 30ms). It has been disabled to protect the imaging pipeline. Reduce the handler's per-frame cost and re-register, or restart the application.
```

The plugin's handler stops firing. The plugin module remains loaded (its `register(ctx)` and `unregister(ctx)` are not called again); the user can manually re-register by calling `ctx.plugins.live_processing.register(...)` again after fixing the cost.

## Common pitfalls

### Don't mutate the supplied frame array

Already covered above. If you take only one rule away, it's this one. The driver's frame array is shared across handlers + display + capture; mutating it makes the next consumer see garbage. Always `image.copy()` if you need to keep it.

### Don't touch Kivy widgets from the handler

The handler runs on the camera SDK thread. Kivy's widget tree is not thread-safe. If you must update a widget from a handler, schedule it on Kivy's main loop:

```python
from kivy.clock import Clock

def handler(image, ts, chunks):
    score = compute_focus_score(image)
    Clock.schedule_once(lambda dt: _update_label(score), 0)
```

### Don't acquire main-thread locks

If your handler waits for a lock that the main thread holds (e.g. via a Kivy property setter that takes a UI lock), you'll deadlock the camera-pump thread. Heavy I/O work belongs on an executor:

```python
from modules.app_context import _app_ctx
from modules.sequential_io_executor import IOTask

def handler(image, ts, chunks):
    # Just enqueue; don't do the work here.
    _app_ctx.ctx.camera_executor.put(
        IOTask(_heavy_save, kwargs={'image': image.copy(), 'ts': ts})
    )
```

### Threading off your own work is fine, but YOU own the lifecycle

If you spawn a `threading.Thread`, log it in `docs/LIFECYCLE_INVENTORY.md` per Rule 41. Your `unregister(ctx)` is responsible for shutting the thread down. If you skip this, the app shutdown hangs.

### The budget is per-invocation, not per-second

24 ms per call. Not 24 ms averaged over a second. If your work is bursty (e.g. one expensive call per 100 cheap calls), you'll get one WARNING for the expensive call but the counter resets on the next cheap call. K=30 consecutive matters; bursty patterns are fine.

### `register` is idempotent on the same handler object

If you call `add_frame_listener(my_handler)` twice with the same function reference, the second call is a no-op. The plugin-namespace path `ctx.plugins.live_processing.register(spec, handler)` raises `PluginRegistrationError` on duplicate `spec.name` instead -- catch that or unregister first.

## Reference

### `scope.imaging.add_frame_listener(cb, name=None)`

Register a per-frame handler.

- `cb`: callable with signature `cb(image, timestamp, chunks)`.
- `name`: display name used in WARNING logs + auto-remove notification. Defaults to the callable's `__qualname__`.

Idempotent for the same `cb`. No-op when no camera is connected.

### `scope.imaging.remove_frame_listener(cb)`

Unregister a handler previously passed to `add_frame_listener`. No-op if the handler wasn't registered.

### `ctx.plugins.live_processing.register(spec, handler)`

Register via the namespace registry. Thin proxy: forwards to `scope.imaging.add_frame_listener(handler, name=spec.name)` and tracks `spec.name -> handler` for unregister-by-name. Raises `PluginRegistrationError` on duplicate `spec.name`.

### `ctx.plugins.live_processing.unregister(name)`

Forward to `scope.imaging.remove_frame_listener(handler)` using the stored name -> handler mapping. No-op if not registered.

### `ctx.plugins.live_processing.names()`

Tuple of currently-registered plugin names. Snapshot semantics -- subsequent register/unregister calls don't appear in a previously-returned tuple.

## Cross-references

- `LumaViewPro/docs/LumascopeSkills.md` -- "Live frame listeners" subsection under `scope.imaging`. Public-facing entry point for external SDK consumers.
- `docs/PERFORMANCE_BUDGETS.md` -- `plugin_live_processing_handler_ms` row. Cites the 24 ms + K=30 constants.
- `docs/WAVE7_PHASE_4D5_PLAN.md` -- the phase plan + section 9 alignment record locking the 24 ms / K=30 / share-with-don't-mutate / warning-severity decisions.
- `PLUGIN_API_DESIGN_2026-05-09.md` -- section 4.5 `live_processing` namespace spec.
- `docs/FRAME_VALIDITY_RIG_COMPARISON_2026-05-19.md` -- the rig data that motivated the 30 fps anchor for the budget.
- `LumaViewPro/modules/lumascope_api/imaging.py` -- `_BudgetedHandler` + the register / remove implementation.
- `LumaViewPro/modules/plugins/__init__.py` -- `LiveProcessingRegistry` thin proxy.
- `LumaViewPro/tests/test_imaging_frame_listener.py` -- 10 tests covering the budget wrapper + end-to-end fan-out + plugin-namespace path.
