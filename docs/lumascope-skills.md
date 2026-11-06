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
> support**, so we can consult you before a structural change. From `4.0.0` the
> API is frozen for the 4.x versions, and every change is recorded in the
> [Changelog](#changelog). What freezes is what the
> [API reference](api-reference.md) lists, and the plugin contract
> ([Plugins](#plugins)), with one exception: the simulator's surface,
> [`SimulationAPI`](api-reference.md#SimulationAPI), may change in any 4.x
> release ([The simulator](guide/simulator.md#the-simulator)). Etaluma's bench and support
> members are not in the reference and are outside the freeze.

**Contents.** [What LumaViewPro is](#what-lumaviewpro-is) ·
[Supported Lumascopes](#supported-lumascopes) · [Examples](#examples) ·
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

## Supported Lumascopes

Every shipped Lumascope is supported. The models differ in what they have, and
the API tells your program what the attached scope has, so it never needs to
know models by name:

```python
caps = scope.capabilities                   # read once, when the scope came up
print(caps.axes, caps.has_xy_stage, caps.has_turret)
print(caps.camera_model, caps.pixel_size_um, caps.led_max_ma)
```

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
  motor controller, and a Basler or IDS camera.
- **FX2 models** have one USB device that serves both the camera and the LEDs.
  The LS720 adds a TMCM-6110 motor board for X, Y and focus; the LS460, LS560
  and LS620 have no motion, and present identically, so the user selects which.

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

Every example, script and twin, and every Python block and the REST
walkthrough in this guide, runs on the simulator in LumaViewPro's test suite.

## The map

Load this page, then the files the task names; each links to the others where
it needs them. Every member, field and outcome is in the
[API reference](api-reference.md), which the files link into by anchor.

| Task | Load |
|---|---|
| Take a still, in Python | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md), [python.md](guide/python.md) |
| Take a still or download a file, over REST | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md), [rest.md](guide/rest.md) |
| Run a scan or a protocol, in Python | [concepts.md](guide/concepts.md), [runs.md](guide/runs.md), [python.md](guide/python.md) |
| Run a scan or a protocol and download its files, over REST | [concepts.md](guide/concepts.md), [runs.md](guide/runs.md), [rest.md](guide/rest.md) |
| Move the stage, the focus or the turret, in Python | [concepts.md](guide/concepts.md), [motion.md](guide/motion.md), [python.md](guide/python.md) |
| Move the stage, the focus or the turret, over REST | [concepts.md](guide/concepts.md), [motion.md](guide/motion.md), [rest.md](guide/rest.md) |
| Light a channel or set the camera, in Python | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [python.md](guide/python.md) |
| Light a channel or set the camera, over REST | [concepts.md](guide/concepts.md), [light-and-camera.md](guide/light-and-camera.md), [rest.md](guide/rest.md) |
| Learn the REST API from a worked session with `curl` | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md), [rest-walkthrough.md](guide/rest-walkthrough.md) |
| Start a REST server, on this computer or for a client on another | [concepts.md](guide/concepts.md), [rest.md](guide/rest.md), [rest-serving.md](guide/rest-serving.md) |
| Test against the simulator or inject a fault, in Python | [concepts.md](guide/concepts.md), [simulator.md](guide/simulator.md), [python.md](guide/python.md) |
| Test against the simulator or inject a fault, over REST | [concepts.md](guide/concepts.md), [simulator.md](guide/simulator.md), [rest.md](guide/rest.md) |
| Drive the scope from another acquisition engine, in Python | [concepts.md](guide/concepts.md), [outside-engines.md](guide/outside-engines.md), [light-and-camera.md](guide/light-and-camera.md), [python.md](guide/python.md) |
| Drive the scope from another acquisition engine, over REST | [concepts.md](guide/concepts.md), [outside-engines.md](guide/outside-engines.md), [light-and-camera.md](guide/light-and-camera.md), [rest.md](guide/rest.md) |

## Plugins

A plugin extends LumaViewPro: a post-processor, a live frame processor, a
panel in the window. It is a Python package loaded at startup, handed the
session, and it uses the API this guide describes. How to write one, its
lifecycle, and which plugin API level a host offers are in
[`plugin-tutorial.md`](plugin-tutorial.md). The plugin contract -- the session's
`plugins` and the namespaces' members the tutorial teaches -- freezes with the
API at `4.0.0`; registering REST routes (`plugins.rest.register`) is outside it.

## Changelog

Every change to the API after `4.0.0`, recorded in the same commit as the
change. Before `4.0.0` the API is fluid by design, and nothing is recorded here.

| Version | Date | Type | Member | Change |
|---|---|---|---|---|

- **Type** `additive` (a new optional argument, field or member): no version
  bump beyond the patch.
- **Type** `behavior-change`: a minor version bump.
- **Type** `rename` / `removal`: a deprecation cycle (the old name kept with a
  `FutureWarning`, removed in the next major version), or a major version bump.

(No entries yet: 4.0.0 has not shipped.)
