#!/usr/bin/env bash
# Holding the scope in a control session, over REST: control_session.py's
# session, on a simulated LumaViewPro serving the API -- the installed app or
# `python -m rest --simulate`; how to start one and run this: README.md here.
# Work in the session carries its id in the Control-Session header.
# Each line's comment is the status it answers, and for a problem its reason.

API=http://127.0.0.1:8000/api/v1

CONTROL=$(curl -s --json '{"name": "plate reader"}' $API/take_control | jq -r .)   # 200: the id, for this client alone
curl -s $API/control                                        # 200: who holds it, and since when

# In the session: a move, and a still.
curl -s -H "Control-Session: $CONTROL" -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200
curl -s -H "Control-Session: $CONTROL" --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0
JOB=$(curl -s -H "Control-Session: $CONTROL" --json '{"false_color_on": false}' $API/manual_capture/capture | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$JOB                 # 200: the still
curl -s -H "Control-Session: $CONTROL" --json '{}' $API/scope/illumination/leds_off   # 200

# Any other client's write is refused while the session is held.
curl -s -H 'Prefer: wait=60' --json '{"z_um": 4000.0}' $API/scope/motion/move_z   # 409 control_session_held

# Ending it; its id then acts no more.
curl -s --json "{\"session_id\": \"$CONTROL\"}" $API/end_control                    # 200
curl -s -H "Control-Session: $CONTROL" -H 'Prefer: wait=60' --json '{"z_um": 4000.0}' $API/scope/motion/move_z   # 422 control_session_ended
