#!/usr/bin/env bash
# Refusals, and a fault the simulator injects, over REST: refusals_and_faults.py's,
# on a simulated LumaViewPro serving the API -- the installed app or
# `python -m rest --simulate`; how to start one and run this: README.md here.
# The server has already homed the scope, so the refusal of a move before
# homing is not shown here.
# Each line's comment is the status it answers, and for a problem its reason.

API=http://127.0.0.1:8000/api/v1

# The request: a move past the stage's travel is 422, whatever the scope's state.
curl -s -H 'Prefer: wait=60' --json '{"z_um": 1000000.0}' $API/scope/motion/move_z   # 422 position_out_of_range

# The camera's cable pulled, by the simulator.
curl -s --json '{}' $API/scope/simulation/remove_camera     # 200
curl -s $API/scope/unconnected_parts                        # 200: ["camera"]

# A still is answered 202 with its job; the job fails with the problem.
curl -s --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0
JOB=$(curl -s --json '{"false_color_on": false}' $API/manual_capture/capture | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$JOB                 # 200: "status": "failed", its "error" not_connected
curl -s --json '{}' $API/scope/illumination/leds_off        # 200
