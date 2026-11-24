#!/usr/bin/env bash
# Holding the scope in a control session, over REST: control_session.py's
# session, on a simulated server running on a copy of LumaViewPro's data/;
# how to start one and run this: README.md here.
# Work in the session carries its id in the Control-Session header.
# Each line's comment is the status it answers, and for a problem its reason.

set -euo pipefail

# The server: this computer's, on the port LVP_PORT names (8000 as shipped).
API="http://127.0.0.1:${LVP_PORT:-8000}/api/v1"
# A job's read answers 200 however the job ended, so its end is checked:
# this passes the job on, or fails naming how it ended.
COMPLETED='if .status == "completed" then . else error("the job ended \(.status): \(.error)") end'
# A refusal shown on purpose: curl fails on it and the script goes on, so the
# answer is checked to be the refusal named, or the script fails.
REFUSAL='if .reason == $reason then . else error("not refused \($reason): \(.)") end'

# A new install has not been told which objective is mounted, and a saved
# still and a run are refused until it has. Confirm the objective the question
# proposes for the slot in the light path (null without a turret); once the
# scope has been told, it proposes what it was told, and this changes nothing.
OBJECTIVE=$(curl --fail-with-body -sS $API/current_slot_objective_question | jq -r .proposed)   # 200
SLOT=$(curl --fail-with-body -sS $API/current_slot_objective_question | jq -r .turret_slot)    # 200: null without a turret
curl --fail-with-body -sS -w '\n' --json "{\"objective_id\": \"$OBJECTIVE\", \"slot\": $SLOT}" $API/confirm_objective   # 200

CONTROL=$(curl --fail-with-body -sS --json '{"name": "plate reader"}' $API/take_control | jq -r .)   # 200: the id, for this client alone
curl --fail-with-body -sS -w '\n' $API/control                                        # 200: who holds it, and since when

# In the session: a move, and a still.
MOVE=$(curl --fail-with-body -sS -H "Control-Session: $CONTROL" --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' -H "Control-Session: $CONTROL" --json '{"channel": "BF"}' $API/show_channel      # 200
curl --fail-with-body -sS -w '\n' -H "Control-Session: $CONTROL" --json '{"lit": true}' $API/light_shown_channel   # 200
JOB=$(curl --fail-with-body -sS -H "Control-Session: $CONTROL" --json '{}' $API/manual_capture/start_save_still | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$JOB | jq -c "$COMPLETED"   # 200: the still
curl --fail-with-body -sS -w '\n' -H "Control-Session: $CONTROL" --json '{"lit": false}' $API/light_shown_channel   # 200

# Any other client's write is refused while the session is held.
ANSWER=$(curl --fail-with-body -sS --json '{"z_um": 4000.0}' $API/scope/motion/start_move_z || true)   # 409 control_session_held
jq -ce --arg reason control_session_held "$REFUSAL" <<< "$ANSWER"

# Ending it; its id then acts no more.
curl --fail-with-body -sS -w '\n' --json "{\"session_id\": \"$CONTROL\"}" $API/end_control                    # 200
ANSWER=$(curl --fail-with-body -sS -H "Control-Session: $CONTROL" --json '{"z_um": 4000.0}' $API/scope/motion/start_move_z || true)   # 422 control_session_ended
jq -ce --arg reason control_session_ended "$REFUSAL" <<< "$ANSWER"
