# API examples

Each example does one task on a simulated microscope, so it needs no hardware
and moves nothing real. Each task comes in two forms that do the same thing:

- `<task>.py` -- Python, calling the API in its own process.
- `<task>.sh` -- the same task over REST, with `curl`, against a simulated
  server running on a copy of LumaViewPro's `data/`.

| Task | Python | REST |
|---|---|---|
| A first saved still | [`first_still.py`](first_still.py) | [`first_still.sh`](first_still.sh) |
| Channels with their own current, exposure and gain | [`channel_settings.py`](channel_settings.py) | [`channel_settings.sh`](channel_settings.sh) |
| A scan of two wells, its images kept | [`scan.py`](scan.py) | [`scan.sh`](scan.sh) |
| Stopping a run | [`stop_a_run.py`](stop_a_run.py) | [`stop_a_run.sh`](stop_a_run.sh) |
| A saved z-stack | [`z_stack_saved.py`](z_stack_saved.py) | [`z_stack_saved.sh`](z_stack_saved.sh) |
| Refusals, and a fault the simulator injects | [`refusals_and_faults.py`](refusals_and_faults.py) | [`refusals_and_faults.sh`](refusals_and_faults.sh) |
| Holding the scope in a control session | [`control_session.py`](control_session.py) | [`control_session.sh`](control_session.sh) |

What each one does and prints is at the top of its file. How the API fits
together is the guide, [`lumascope-skills.md`](../lumascope-skills.md); every
member is in [`api-reference.md`](../api-reference.md). LumaViewPro's test
suite runs every example, both forms, and checks what it leaves.

## Running a Python example

A Python example needs LumaViewPro's source: it imports the API from
`lumascope`, which the installed app keeps inside its program file.

1. Get the source and install its Python packages, as LumaViewPro's
   [README](https://github.com/EtalumaSupport/LumaViewPro#getting-started) says (Python 3.12 or
   3.13). Its install scripts also make `lumascope` importable by the Python
   they installed into, and that Python runs the examples.
2. Open a terminal in the LumaViewPro folder, the one holding
   `lumaviewpro.py`. If you installed into a virtual environment, activate it.
3. Run the example by its path:

   ```
   python docs/api_examples/first_still.py
   ```

   (`python3` on macOS and Linux.)

It prints what it did. Its images are saved in the live folder your
LumaViewPro settings name: `capture/` in a fresh copy of the source.

## Running a REST example

A REST example runs only against a simulated server started from the source
on a copy of its `data/` folder, never against LumaViewPro's own settings.
Each example confirms the objective the simulator proposes and writes the
settings it uses; on your own `data/`, a scope with no turret would then
never be asked which objective is mounted, and its files would carry the
default objective's scale, whatever is mounted. A copy keeps every write and capture out of your files.

You need LumaViewPro's source, installed as in [Running a Python
example](#running-a-python-example), and `bash`, `curl` 7.82 or newer and
[`jq`](https://jqlang.github.io/jq/). On Windows, run them in Git Bash (from
[Git for Windows](https://gitforwindows.org)), which has `bash` and `curl`
(install `jq` with `winget install jqlang.jq`), or in WSL with the source
installed there; run both steps below in the same one.

1. Start the simulated server on a copy of `data/`. In the LumaViewPro
   folder, the one holding `lumaviewpro.py`:

   ```
   mkdir -p /tmp/lvp-sim && cp -R data /tmp/lvp-sim/ && rm -f /tmp/lvp-sim/data/current.json
   python -m lumascope.rest --simulate --source-path /tmp/lvp-sim
   ```

   (`python3` on macOS and Linux.) The copy has no `current.json`, so the
   server comes up as a new install. The scope homes as it starts; once it
   has, the server prints `LumaViewPro REST server:` and where it serves. It
   serves until you press Ctrl-C.
2. In another terminal, in a folder you can write to (`first_still.sh` and
   `scan.sh` download an image into it), run the example by its path:

   ```
   bash <LumaViewPro folder>/docs/api_examples/first_still.sh
   ```

Each line prints its answer, and its comment says the status it answers.
An example stops at the first answer it does not expect, with the problem
it was answered printed, and exits non-zero; one that shows a refusal on
purpose checks that it is that refusal and goes on, and `curl` prints the
refusal's status as it does. The server is `http://127.0.0.1:8000`, this
computer only. A copy of `data/` with another `rest_api.port` serves on
that port: name it to the example with `LVP_PORT`, as in
`LVP_PORT=8001 bash .../first_still.sh`. The interactive reference of
every route is at `http://127.0.0.1:8000/docs` while the server runs.

`refusals_and_faults` pulls the simulated camera's cable, as its Python form
does: restart the server after it to have a camera again.
