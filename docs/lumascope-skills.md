# The Lumascope API guide

This guide shows how to drive an Etaluma Lumascope from your own program, in
Python or over REST: how the API fits together, and how to do each task with
it. Every member, field, closed set and outcome class is in the
[API reference](api-reference.md), generated from the code; this guide links to
it rather than repeating it. The guide is this page and the files under
[`guide/`](guide): load the ones [the map](#the-map) names for your task.

> **Pre-release.** Until LumaViewPro `4.0.0` ships, the API may change without a
> deprecation cycle: members, routes, records and outcome reasons may be
> renamed, moved or removed. Most likely to move are the REST surface (its
> routes, handles, jobs, problems and event stream) and the protocol members
> and file columns. If you build on the API before `4.0.0`, **contact Etaluma
> support**, so we can consult you before a structural change. From `4.0.0` a
> 4.x release changes the API only as the [Changelog](#changelog)'s rules
> allow -- an addition or a behaviour change at a minor version, never a rename
> or removal without a deprecation cycle -- and records each change there. What freezes is what the
> [API reference](api-reference.md) lists, and the plugin contract
> ([Plugins](#plugins)), with one exception: the simulator's surface,
> [`SimulationAPI`](api-reference.md#SimulationAPI), may change in any 4.x
> release ([The simulator](guide/simulator.md#the-simulator)). Etaluma's bench and support
> members are not in the reference and are outside the freeze.

**Contents.** [What LumaViewPro is](#what-lumaviewpro-is) ·
[Supported Lumascopes](#supported-lumascopes) ·
[The architecture at a glance](#the-architecture-at-a-glance) ·
[The API at a glance](#the-api-at-a-glance) · [Examples](#examples) ·
[The map](#the-map) · [Plugins](#plugins) · [Changelog](#changelog)

## What LumaViewPro is

LumaViewPro controls Etaluma's Lumascope microscopes: the LEDs that light the
sample, the stage, focus and objective turret that move it, and the camera that
images it. It runs as a desktop application, and the API it is built on is open
to your own programs:

- **Python**, in the same process: a script, a notebook, MATLAB calling Python,
  or a plugin loaded into LumaViewPro.
- **REST**, over HTTP from any language: a server LumaViewPro starts beside its
  window, or one you start without a window at all.

This guide is for researchers who script an experiment and for developers who
integrate a Lumascope into a larger system. LumaViewPro runs on Python 3.12 and
3.13, on Windows, macOS and Linux.

A Python program needs LumaViewPro's source: the installed app keeps the API
inside its program file. LumaViewPro's install scripts make `lumascope`
importable by the Python they install into
([README](https://github.com/EtalumaSupport/LumaViewPro#getting-started)), and
that Python runs your program. A REST client needs only a running server.
To start, bring a session up on the simulator, which needs no microscope
([Bringing a session up in Python](guide/python.md#bringing-a-session-up-in-python),
`simulate=True`), or start a simulated server
([Serving it](guide/rest.md#serving-it), `--simulate`). Where your settings,
captures and logs live is in the same two places.

## Supported Lumascopes

Every shipped Lumascope is supported. The models differ in what they have, and
the API tells your program what the attached scope has, so it never needs to
know models by name. Here, as in every example, `session` is the
`ScopeSession` your program brought up and `scope` is `session.scope`:

```python
caps = scope.capabilities                   # read once, when the scope came up
print(caps.axes, caps.has_xy_stage, caps.has_turret)
print(caps.camera_model, caps.pixel_size_um, caps.led_max_ma)
```

Over REST it is `GET /api/v1/scope/capabilities`. `camera_model` names the
sensor, not the scope: every FX2 model answers `MT9P031-LS620`.

| Model | Focus (Z) | XY stage | Turret | Channels | Controller |
|---|---|---|---|---|---|
| LS820 | yes | -- | -- | BF, PC, DF, Ex405, Ex488, Ex589 | EL-0940 |
| LS850 | yes | yes | -- | BF, PC, DF, Ex405, Ex488, Ex589 | EL-0940 |
| LS850-0 | yes | yes | -- | BF | EL-0940 |
| LS850T | yes | yes | yes | BF, PC, DF, Ex405, Ex488, Ex589 | EL-0940 |
| LS850T-0 | yes | yes | yes | BF, PC | EL-0940 |
| Lumi | yes | -- | -- | BF, PC, DF, Lumi | EL-0940 |
| LS560 | -- | -- | -- | BF, Ex488 | FX2 |
| LS460 | -- | -- | -- | BF | FX2 |
| LS620 | -- | -- | -- | BF, Ex405, Ex488, Ex589 | FX2 |
| LS720 | yes | yes | -- | BF, Ex405, Ex488, Ex589 | FX2, with a TMCM-6110 motor board |

- **Channels.** `BF` brightfield, `PC` phase contrast, `DF` darkfield, `Ex405`,
  `Ex488` and `Ex589` fluorescence named for its excitation wavelength in nm, and
  `Lumi` luminescence, which lights no LED. The FX2 models label brightfield
  "BF-Phase". The release also knows `Ex550` and `Ex640`. Which channels the
  attached scope has, and which LEDs each drives, is `scope.channel_identity`
  ([Illumination](guide/light-and-camera.md#illumination)).
- **EL-0940 models** have one mainboard carrying the LED controller and the
  motor controller, and a Basler or IDS camera, either on any model
  (`scope.capabilities.camera_model` says which).
- **FX2 models** have one USB device that serves both the camera and the LEDs.
  The LS720 adds a TMCM-6110 motor board for X, Y and focus; the LS460, LS560
  and LS620 have no motion, and present identically, so the user selects which.

## The architecture at a glance

```text
 Your program                       LumaViewPro's window
 (Python script, REST client)       (one client of the same API)
        |                                      |
        v                                      |
   REST server  --- one route per member ---   |
        |                                      |
        v                                      v
 ScopeSession -- settings, protocols, runs, the control session, outcomes
   v
 Lumascope (session.scope) -- the instrument, one area per part
   v
 Drivers -- the motor board, the LED board, the camera
   v
 Hardware (or the simulator, which stands in for every board)
```

- **The session**, `ScopeSession`, one per microscope, is brought up by
  `ScopeSession.create` the same way in every host, and owns the settings, runs,
  captures, who may write, and outcomes. **The scope**, `session.scope`, is the
  instrument, in the areas below.
- **Python and REST are one API**: `scope.motion.move_z` is
  `POST /api/v1/scope/motion/move_z`, with the same refusals, faults and
  events. Each part's commands, from every caller, run one at a time, in order.
- **A call that works returns its value; every other outcome is a typed
  exception** with a `reason`, raised to the caller or posted to listeners: a
  refusal (nothing was done), a fault (something broke) or a notice (something
  the caller should know, a stopped move among them).
- **One client at a time may hold the scope** in a control session; a run, a
  home or a recording holds it for its length.

The detail is [concepts.md](guide/concepts.md).

## The API at a glance

| Area | What it does, and the members that matter | File |
|---|---|---|
| Bring-up and settings | `ScopeSession.create` (`simulate=True` for the simulator), then `session.run_application_session()` to home; `session.update_settings` writes a setting by path. | [concepts](guide/concepts.md), [python](guide/python.md) |
| `scope.motion` | Homes and moves the stage, focus and turret: `home`, `move_z`, `move_xy`, `move_to_well`, `move_turret`, `start_move_z`; `get_axis_limits` bounds every move. | [motion](guide/motion.md) |
| `scope.illumination`, `scope.camera` | The LEDs (`led_on`, `leds_off`), the camera's settings (`set_exposure_ms`, `set_gain_db`), and `session.show_channel`, the channel a capture is named for. | [light-and-camera](guide/light-and-camera.md) |
| Captures and files | A saved still (`session.manual_capture.capture`), a recording (`session.manual_recording.start`), and the record a saved file carries. | [captures-and-files](guide/captures-and-files.md) |
| Protocols and runs | A protocol of steps (`create_empty_protocol`, `add_step`), run by `session.protocol_runner()` (`start_single_scan`, `start_protocol`), then post-processed (`post_processing.build`). | [runs](guide/runs.md) |
| `scope.simulation` | Faults injected into a simulated scope. | [simulator](guide/simulator.md) |
| The control session | `take_control`, `acting_in_control`, `end_control`; over REST, the `Control-Session` header. | [concepts](guide/concepts.md), [rest](guide/rest.md#the-control-session) |
| REST's own routes | Serving it, jobs, handles, problems, events, the live view and files, a walkthrough with `curl`, a client on another computer. | [rest](guide/rest.md) |
| In-process Python | Operations, outcome listeners, live frames and events, a still as an image. | [python](guide/python.md) |
| Another acquisition engine | What Micro-Manager, pymmcore-plus or a lab's own sequencer needs to drive the scope as it sets it. | [outside-engines](guide/outside-engines.md) |

## Examples

[`api_examples/`](api_examples) holds a Python script per task, each with a
`.sh` twin that does the same over REST with `curl`. Its
[README](api_examples/README.md) says how to run them: a script from
LumaViewPro's source, a twin against the installed app or
`python -m lumascope.rest --simulate`.

| Example | What it shows |
|---|---|
| `first_still` | a first saved still |
| `channel_settings` | channels with their own current, exposure and gain |
| `scan` | a scan whose images are kept |
| `stop_a_run` | stopping a run |
| `z_stack_saved` | a saved z-stack |
| `refusals_and_faults` | refusals and an injected fault |
| `control_session` | the control session |

Every example, script and twin, every Python block of this guide but the
server in [Serving it](guide/rest.md#serving-it), which serves until it is
stopped, and the REST walkthrough run on the simulator in LumaViewPro's test
suite.

## The map

Load this page, then the files the task names; each links to the others where
it needs them. Every member, field and outcome is in the
[API reference](api-reference.md), which the files link into by anchor.

| Task | Load |
|---|---|
| Bring a session up, or read and write its settings, in Python | [concepts.md](guide/concepts.md), [python.md](guide/python.md) |
| Read and write the settings, over REST | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md) |
| Take a still, in Python | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md), [python.md](guide/python.md) |
| Take a still or download a file, over REST | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md), [rest.md](guide/rest.md) |
| Record a video, in Python | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md), [python.md](guide/python.md) |
| Record a video, over REST | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md), [rest.md](guide/rest.md) |
| Run a scan or a protocol, in Python | [concepts.md](guide/concepts.md), [runs.md](guide/runs.md), [python.md](guide/python.md) |
| Run a scan or a protocol and download its files, over REST | [concepts.md](guide/concepts.md), [runs.md](guide/runs.md), [rest.md](guide/rest.md) |
| Build a stack, stitch, composite or video from a run's folder, in Python | [concepts.md](guide/concepts.md), [runs.md](guide/runs.md), [python.md](guide/python.md) |
| Build a stack, stitch, composite or video from a run's folder, over REST | [concepts.md](guide/concepts.md), [runs.md](guide/runs.md), [rest.md](guide/rest.md) |
| Move the stage, the focus or the turret, in Python | [concepts.md](guide/concepts.md), [motion.md](guide/motion.md), [python.md](guide/python.md) |
| Move the stage, the focus or the turret, over REST | [concepts.md](guide/concepts.md), [motion.md](guide/motion.md), [rest.md](guide/rest.md) |
| Light a channel or set the camera, in Python | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [python.md](guide/python.md) |
| Light a channel or set the camera, over REST | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [rest.md](guide/rest.md) |
| Learn the REST API from a worked session with `curl` | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md) |
| Start a REST server, on this computer or for a client on another | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md) |
| Hear live frames or events, in Python | [concepts.md](guide/concepts.md), [python.md](guide/python.md) |
| Hear events or show the live view, over REST | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md) |
| Hold the scope in a control session, in Python | [concepts.md](guide/concepts.md), [python.md](guide/python.md) |
| Hold the scope in a control session, over REST | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md) |
| Test against the simulator or inject a fault, in Python | [concepts.md](guide/concepts.md), [simulator.md](guide/simulator.md), [python.md](guide/python.md) |
| Test against the simulator or inject a fault, over REST | [concepts.md](guide/concepts.md), [simulator.md](guide/simulator.md), [rest.md](guide/rest.md) |
| Drive the scope from another acquisition engine, in Python | [concepts.md](guide/concepts.md), [outside-engines.md](guide/outside-engines.md), [light-and-camera.md](guide/light-and-camera.md), [python.md](guide/python.md) |
| Drive the scope from another acquisition engine, over REST | [concepts.md](guide/concepts.md), [outside-engines.md](guide/outside-engines.md), [light-and-camera.md](guide/light-and-camera.md), [rest.md](guide/rest.md) |
| Write a plugin | [plugin-tutorial.md](plugin-tutorial.md), [python.md](guide/python.md) |

## Plugins

A plugin extends LumaViewPro: a post-processor, a live frame processor, a
panel in the window. It is a Python package loaded at startup, handed the
session, and it uses the API this guide describes. How to write one, its
lifecycle, and which plugin API level a host offers are in
[`plugin-tutorial.md`](plugin-tutorial.md). The plugin contract -- the session's
`plugins` and the namespaces' members the tutorial teaches -- freezes with the
API at `4.0.0`. A plugin cannot add REST routes in this release
(`plugins.rest.register` is refused `'rest_namespace_reserved'`), and that
namespace is outside the freeze.

## Changelog

Every change to the API after `4.0.0`, recorded in the same commit as the
change. Before `4.0.0` the API is fluid by design, and nothing is recorded here.

| Version | Date | Type | Member | Change |
|---|---|---|---|---|

- **Type** `additive` (a new optional argument, field or member): a minor
  version bump (4.x.0); a patch bump is fixes only.
- **Type** `behavior-change`: a minor version bump.
- **Type** `rename` / `removal`: a deprecation cycle (the old name kept with a
  `FutureWarning`, removed in the next major version), or a major version bump.

(No entries yet: 4.0.0 has not shipped.)
