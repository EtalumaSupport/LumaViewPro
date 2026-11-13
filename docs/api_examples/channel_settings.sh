#!/usr/bin/env bash
# Channels with their own current, exposure and gain, over REST:
# channel_settings.py's stills, on a simulated LumaViewPro serving the API --
# the installed app or `python -m lumascope.rest --simulate`; how to start one and run
# this: README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

API=http://127.0.0.1:8000/api/v1

# A new install has not been told which objective is mounted, and a saved
# still and a run are refused until it has. Confirm the objective the question
# proposes for the slot in the light path (null without a turret); once the
# scope has been told, it proposes what it was told, and this changes nothing.
OBJECTIVE=$(curl -s $API/current_slot_objective_question | jq -r .proposed)   # 200
SLOT=$(curl -s $API/current_slot_objective_question | jq -r .turret_slot)    # 200: null without a turret
curl -s -w '\n' --json "{\"objective_id\": \"$OBJECTIVE\", \"slot\": $SLOT}" $API/confirm_objective   # 200

MOVE=$(curl -s --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl -s -w '\n' -H 'Prefer: wait=60' $API/jobs/$MOVE               # 200: "status": "completed"

# Brightfield's settings, in one write; a key left out is left as it is.
# Showing the channel puts them on the camera.
curl -s -w '\n' -X PUT --json '{"channel": "BF", "illumination_ma": 5, "exposure_ms": 2.0, "gain_db": 1.0}' $API/channel_settings   # 200: BF's settings
curl -s -w '\n' --json '{"channel": "BF"}' $API/show_channel                                 # 200
curl -s -w '\n' --json '{"lit": true}' $API/light_shown_channel                              # 200
BF_JOB=$(curl -s --json '{}' $API/manual_capture/start_save_still | jq -r .id)   # 202
curl -s -w '\n' -H 'Prefer: wait=60' $API/jobs/$BF_JOB              # 200: the still, its frame held for BF
curl -s -w '\n' --json '{"lit": false}' $API/light_shown_channel                             # 200

# The same for a fluorescence channel.
curl -s -w '\n' -X PUT --json '{"channel": "Ex488", "illumination_ma": 300, "exposure_ms": 100.0, "gain_db": 10.0}' $API/channel_settings   # 200
curl -s -w '\n' --json '{"channel": "Ex488"}' $API/show_channel                                 # 200
curl -s -w '\n' --json '{"lit": true}' $API/light_shown_channel                                 # 200
EX488_JOB=$(curl -s --json '{}' $API/manual_capture/start_save_still | jq -r .id)   # 202
curl -s -w '\n' -H 'Prefer: wait=60' $API/jobs/$EX488_JOB           # 200
curl -s -w '\n' --json '{"lit": false}' $API/light_shown_channel                                # 200

# A still is of the shown channel, lit by its own LED alone. With brightfield
# shown and lit, a raw camera write stands for the next still, which holds no
# channel's settings.
curl -s -w '\n' --json '{"channel": "BF"}' $API/show_channel                         # 200
curl -s -w '\n' --json '{"lit": true}' $API/light_shown_channel                      # 200
curl -s -w '\n' -X PUT --json '{"exposure_ms": 1.5}' $API/scope/camera/exposure_ms   # 200: 1.5
curl -s -w '\n' -X PUT --json '{"gain_db": 3.0}' $API/scope/camera/gain_db           # 200: 3.0
RAW_JOB=$(curl -s --json '{}' $API/manual_capture/start_save_still | jq -r .id)   # 202
curl -s -w '\n' -H 'Prefer: wait=60' $API/jobs/$RAW_JOB             # 200
curl -s -w '\n' --json '{"lit": false}' $API/light_shown_channel    # 200
