#!/usr/bin/env bash
# Channels with their own current, exposure and gain, over REST:
# channel_settings.py's stills, on a server started with
# `python -m rest --simulate`, which has already run the startup motion.
# Each line's comment is the status it answers.

API=http://127.0.0.1:8000/api/v1

curl -s -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200

# Brightfield's settings, by path; showing it puts them on the camera.
curl -s --json '{"path": "BF.illumination_ma", "value": 5}' $API/update_settings     # 200
curl -s --json '{"path": "BF.exposure_ms", "value": 2.0}' $API/update_settings       # 200
curl -s --json '{"path": "BF.gain_db", "value": 1.0}' $API/update_settings           # 200
curl -s --json '{"channel": "BF"}' $API/show_channel                                 # 200
curl -s --json '{"lit": true}' $API/light_shown_channel                              # 200
BF_JOB=$(curl -s --json '{"false_color_on": false}' $API/manual_capture/capture | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$BF_JOB              # 200: the still, its frame held for BF
curl -s --json '{"lit": false}' $API/light_shown_channel                             # 200

# The same for a fluorescence channel.
curl -s --json '{"path": "Ex488.illumination_ma", "value": 300}' $API/update_settings   # 200
curl -s --json '{"path": "Ex488.exposure_ms", "value": 100.0}' $API/update_settings     # 200
curl -s --json '{"path": "Ex488.gain_db", "value": 10.0}' $API/update_settings          # 200
curl -s --json '{"channel": "Ex488"}' $API/show_channel                                 # 200
curl -s --json '{"lit": true}' $API/light_shown_channel                                 # 200
EX488_JOB=$(curl -s --json '{"false_color_on": false}' $API/manual_capture/capture | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$EX488_JOB           # 200
curl -s --json '{"lit": false}' $API/light_shown_channel                                # 200

# A raw camera write stands for the next still, which holds no channel's settings.
curl -s -X PUT --json '{"exposure_ms": 1.5}' $API/scope/camera/exposure_ms   # 200: 1.5
curl -s -X PUT --json '{"gain_db": 3.0}' $API/scope/camera/gain_db           # 200: 3.0
curl -s --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0
RAW_JOB=$(curl -s --json '{"false_color_on": false}' $API/manual_capture/capture | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$RAW_JOB             # 200
curl -s --json '{}' $API/scope/illumination/leds_off        # 200
