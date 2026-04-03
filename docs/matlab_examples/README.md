# MATLAB Integration Examples

These examples show how to control an Etaluma microscope from MATLAB using the LumaViewPro REST API.

## Prerequisites

- MATLAB R2016b+ (for `webread`/`webwrite`)
- LumaViewPro running with REST API enabled
- Microscope connected and homed

## Setup

1. Start LumaViewPro
2. Enable REST API in Settings (or `data/settings.json`: `"rest_api": {"enabled": true}`)
3. The API runs at `http://localhost:8000` by default

## Examples

| File | Description |
|------|-------------|
| `lvp_connect.m` | Helper class — wraps all REST calls |
| `basic_capture.m` | Move, illuminate, capture, save |
| `multi_channel.m` | BF + fluorescence composite |
| `well_plate_scan.m` | Scan across a 96-well plate |
| `z_stack.m` | Z-stack acquisition |
| `timelapse.m` | Repeated captures with interval |
| `autofocus_capture.m` | Autofocus then capture |

## API Status

The REST API is planned for LumaViewPro 4.1. These examples preview the API design. Endpoint signatures may change before release.
