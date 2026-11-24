#!/usr/bin/env bash
# A first saved still, over REST: first_still.py's still, on a simulated
# server running on a copy of LumaViewPro's data/; how to start one and run
# this: README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

set -euo pipefail

# The server: this computer's, on the port LVP_PORT names (8000 as shipped).
API="http://127.0.0.1:${LVP_PORT:-8000}/api/v1"
# A job's read answers 200 however the job ended, so its end is checked:
# this passes the job on, or fails naming how it ended.
COMPLETED='if .status == "completed" then . else error("the job ended \(.status): \(.error)") end'

# A new install has not been told which objective is mounted, and a saved
# still and a run are refused until it has. Confirm the objective the question
# proposes for the slot in the light path (null without a turret); once the
# scope has been told, it proposes what it was told, and this changes nothing.
OBJECTIVE=$(curl --fail-with-body -sS $API/current_slot_objective_question | jq -r .proposed)   # 200
SLOT=$(curl --fail-with-body -sS $API/current_slot_objective_question | jq -r .turret_slot)    # 200: null without a turret
curl --fail-with-body -sS -w '\n' --json "{\"objective_id\": \"$OBJECTIVE\", \"slot\": $SLOT}" $API/confirm_objective   # 200

MOVE=$(curl --fail-with-body -sS --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' --json '{"channel": "BF"}' $API/show_channel                         # 200
curl --fail-with-body -sS -w '\n' --json '{"lit": true}' $API/light_shown_channel                      # 200

# A still is answered 202 with its job; the job's result is the saved files.
JOB=$(curl --fail-with-body -sS --json '{}' $API/manual_capture/start_save_still | jq -r .id)   # 202
FILE=$(curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r "$COMPLETED | .result[0].file.name")   # 200
curl --fail-with-body -sS -o still.tiff "$API/files/$FILE"                    # 200: the TIFF, its record inside
curl --fail-with-body -sS -w '\n' --json '{"lit": false}' $API/light_shown_channel                     # 200
