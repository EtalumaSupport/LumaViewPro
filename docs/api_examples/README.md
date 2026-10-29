# API examples

Each example does one task on a simulated microscope, so it needs no hardware
and moves nothing real. Each task comes in two forms that do the same thing:

- `<task>.py` -- Python, calling the API in its own process.
- `<task>.sh` -- the same task over REST, with `curl`, against a running
  LumaViewPro.

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

A Python example needs LumaViewPro's source: it imports the API, which the
installed app keeps inside its program file.

1. Get the source and install its Python packages, as LumaViewPro's
   [README](https://github.com/EtalumaSupport/LumaViewPro#getting-started) says (Python 3.12 or
   3.13).
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

A REST example runs against a LumaViewPro that is serving the API: the
installed app, or a server started from the source. You need `bash`, `curl`
7.82 or newer and [`jq`](https://jqlang.github.io/jq/). On Windows, Git Bash (from
[Git for Windows](https://gitforwindows.org)) has `bash` and `curl`; install
`jq` with `winget install jqlang.jq`.

1. Start a simulated LumaViewPro that serves the API, one of:
   - **The installed app.** Close LumaViewPro. Open
     `Documents\LumaViewPro <version>\data\current.json` (made the first time
     the app starts) in a text editor and set

     ```json
     "rest_api": {"enabled": true, "port": 8000}
     ```

     Then start the app in simulation from a command prompt, in its folder
     under `Program Files`:

     ```
     lumaviewpro.exe --simulate
     ```

     It serves the API while its window is open. Set `enabled` back to
     `false` when you are done.
   - **From the source.** In the LumaViewPro folder:

     ```
     python -m rest --simulate
     ```

     It serves until you press Ctrl-C.

   Either way the scope homes as it starts; wait for that to finish.
2. In another terminal, in a folder you can write to (`first_still.sh` and
   `scan.sh` download an image into it), run the example by its path: in
   the source, or in the installed app's folder, which carries these
   examples too.

   ```
   bash <LumaViewPro folder>/docs/api_examples/first_still.sh
   ```

Each line prints its answer on a line of its own, and its comment says the
status it answers. The server is
`http://127.0.0.1:8000`, this computer only; with another `port`, change
`API=` at the top of the example. The interactive reference of every route is
at `http://127.0.0.1:8000/docs` while the server runs.

`refusals_and_faults` pulls the simulated camera's cable, as its Python form
does: restart the server after it to have a camera again.
