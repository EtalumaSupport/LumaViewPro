#!/usr/bin/env bash
# Refusals, and a fault the simulator injects, over REST: refusals_and_faults.py's,
# on a simulated server running on a copy of LumaViewPro's data/; how to start
# one and run this: README.md here.
# The server has already homed the scope, so the refusal of a move before
# homing is not shown here.
# Each line's comment is the status it answers, and for a problem its reason.

set -euo pipefail

# The server: this computer's, on the port LVP_PORT names (8000 as shipped).
API="http://127.0.0.1:${LVP_PORT:-8000}/api/v1"
# A refusal shown on purpose: curl fails on it and the script goes on, so the
# answer is checked to be the refusal named, or the script fails.
REFUSAL='if .reason == $reason then . else error("not refused \($reason): \(.)") end'
# A job's read answers 200 however the job ended, so its end is checked:
# this passes the job on if it failed with the problem named, or fails.
FAILED='if .status == "failed" and .error.reason == $reason then . else error("the job did not fail \($reason): \(.)") end'

# The request: a move past the stage's travel is 422, whatever the scope's state.
ANSWER=$(curl --fail-with-body -sS --json '{"z_um": 1000000.0}' $API/scope/motion/start_move_z || true)   # 422 position_out_of_range
jq -ce --arg reason position_out_of_range "$REFUSAL" <<< "$ANSWER"

# The camera's cable pulled, by the simulator.
curl --fail-with-body -sS -w '\n' --json '{}' $API/scope/simulation/remove_camera     # 200
curl --fail-with-body -sS -w '\n' $API/scope/unconnected_parts                        # 200: ["camera"]

# A still is answered 202 with its job; the job fails with the problem.
curl --fail-with-body -sS -w '\n' --json '{"channel": "BF"}' $API/show_channel                         # 200
curl --fail-with-body -sS -w '\n' --json '{"lit": true}' $API/light_shown_channel                      # 200
JOB=$(curl --fail-with-body -sS --json '{}' $API/manual_capture/start_save_still | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$JOB | jq -ce --arg reason not_connected "$FAILED"   # 200: "status": "failed", its "error" not_connected
curl --fail-with-body -sS -w '\n' --json '{"lit": false}' $API/light_shown_channel                     # 200
