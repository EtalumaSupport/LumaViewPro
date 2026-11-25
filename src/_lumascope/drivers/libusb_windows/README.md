# libusb for the Lumascope Classic on Windows x64

The Lumascope Classic (LS460, LS560, LS620, LS720) streams through libusb.
On Windows x64, LumaViewPro runs on the `libusb-1.0.dll` in this folder
and no other; on macOS and Linux it runs on the `libusb-package` wheel's
library (`requirements.txt`).

## Why this build

libusb 1.0.30's WinUSB isochronous path resubmits a failed transfer
itself, without calling the caller's callback, and drops the resubmit's
failure. At every stop of a Classic's stream on Windows about half of
the 16 transfers are lost that way, so the stream's close waits on
transfers that never come back. libusb PR #1967 fixes it: a cancelled,
timed-out or disconnected transfer goes to the caller's callback
unretried, and a retry that fails is reported to it.

## What is here

| File | What it is |
|---|---|
| `0001-windows-ensure-iso-completion-callbacks-on-cancellation-and-retry-failure.patch` | libusb PR #1967 (https://github.com/libusb/libusb/pull/1967), head `12a249cca857860bcff1ad59424d87a7d66ed394`, backported to v1.0.30: one context line changed for a macro renamed on master since the tag |
| `0002-etaluma-build-identity.patch` | Etaluma's: `LIBUSB_RC` is `-etaluma.1`, so the library names itself, and `windows_winusb.c` says it was modified |
| `libusb-1.0.dll` | The x64 build of libusb `v1.0.30` (`87a55632db62c9bdc58cd31d3ccfa673f1bb017f`) with both patches |
| `libusb-1.0.30-etaluma.1-source.zip` | The complete source the DLL was built from (LGPL-2.1 section 4) |

The DLL and the source zip come from one run of
`.github/workflows/build-libusb-windows.yml`, committed by hand from its
artifact.

## Retire-by

The first libusb release that carries #1967, once a `libusb-package`
wheel carries that release: Windows returns to the wheel, and this
folder and the workflow go. Until then, the backport is redone only if
#1967 changes before it merges.
