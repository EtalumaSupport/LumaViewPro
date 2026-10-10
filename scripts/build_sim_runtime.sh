#!/usr/bin/env bash
# Build the MicroPython runtimes the simulated boards run in, from pinned
# upstream source (MIT licensed; no Etaluma firmware is involved).
#
#   scripts/build_sim_runtime.sh            build for this machine
#
# One runtime per MicroPython version in drivers/sim_wire/runtime/MICROPYTHON_PIN
# (the one place the versions live): each firmware dialect runs on the
# version its boards run, because the versions differ in ways the firmware
# depends on (1.19 parses int('0x10') as hex; 1.28 refuses it).
#
# Output, per version:
#   drivers/sim_wire/runtime/<platform>/micropython-<tag>
#       darwin        universal2 (arm64 + x86_64); committed to the repo
#       linux-x86_64  built where it runs (CI, a Linux developer); not committed
#   build/sim_runtime/mpy-cross-<tag>, for compiling firmware to .mpy
# The upstream clones are kept in ~/.cache/lvp-sim-runtime/ (or
# $XDG_CACHE_HOME), outside the repository.
#
# Build choices, all so the runtime behaves as the board does:
# - FFI off: the firmware never uses it and libffi's headers are not on a
#   stock Mac.
# - Ctrl-C is scheduled, not raised from inside the signal handler. Raised
#   there, it lands wherever the VM is (mid-GC, mid-print), and about one
#   Ctrl-C in fifteen killed the runtime with 'FATAL: uncaught NLR'. 1.28
#   gets this from turning the GIL on; 1.19 from its patch.
# - scripts/sim_runtime_<tag>_*.patch: a Ctrl-C that lands just before the
#   firmware blocks reading stdin waits for the next input byte instead of
#   firing; the patch makes it fire, as on the board.
# - 1.19 predates the compilers on current machines, so its warnings are not
#   errors.
# - Floats are single precision, as on the board: the RP2040 port builds
#   MicroPython that way and the unix port defaults to double, so the same
#   firmware arithmetic gives different results (47.92 prints as
#   47.92000000000001 in double).
# - Each unix build starts clean: make does not rebuild objects when only the
#   flags change, so a changed flag would otherwise link the old objects.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_FILE="$ROOT/drivers/sim_wire/runtime/MICROPYTHON_PIN"

case "$(uname -s)-$(uname -m)" in
    Darwin-*) PLATFORM=darwin ;;
    Linux-x86_64) PLATFORM=linux-x86_64 ;;
    *)
        echo "no simulator runtime for $(uname -s)-$(uname -m)" >&2
        exit 1
        ;;
esac
OUT="$ROOT/drivers/sim_wire/runtime/$PLATFORM"
TOOLS="$ROOT/build/sim_runtime"
# The MicroPython clones live outside the repository: they are third-party
# source, including Python that the interpreter running this repo's tools
# and tests cannot parse.
SRC_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/lvp-sim-runtime"
mkdir -p "$OUT" "$TOOLS" "$SRC_CACHE"

# Copy out the unix port binary just built: 1.28 writes it inside the build
# directory, 1.19 beside the Makefile (so it is copied before the next arch
# overwrites it).
unix_binary() {
    local src="$1" build="$2" dest="$3"
    if [ -x "$src/ports/unix/$build/micropython" ]; then
        cp "$src/ports/unix/$build/micropython" "$dest"
    else
        cp "$src/ports/unix/micropython" "$dest"
    fi
}

build_one() {
    local tag="$1" commit="$2"
    local version="${tag#v}"
    local src="$SRC_CACHE/micropython-$version"
    local flags=(MICROPY_PY_FFI=0)
    local cflags=""
    case "$version" in
        1.28.*) flags+=(MICROPY_PY_THREAD_GIL=1) ;;
        1.19.*) cflags="-Wno-error" ;;
        *)
            echo "no build recipe for MicroPython $tag" >&2
            exit 1
            ;;
    esac

    if [ ! -d "$src/.git" ]; then
        git clone --quiet --depth 1 --branch "$tag" https://github.com/micropython/micropython.git "$src"
    fi
    local head
    head="$(git -C "$src" rev-parse HEAD)"
    if [ "$head" != "$commit" ]; then
        echo "$src is at $head, the pin is $tag $commit; remove it and re-run" >&2
        exit 1
    fi

    # The pinned source plus our patches, and nothing else.
    git -C "$src" checkout --quiet -- .
    for patch in "$ROOT"/scripts/sim_runtime_"$tag"_*.patch; do
        [ -e "$patch" ] && git -C "$src" apply "$patch"
    done

    make -C "$src/mpy-cross" -j8 "${flags[@]}" CFLAGS_EXTRA="$cflags" >/dev/null
    if [ -x "$src/mpy-cross/build/mpy-cross" ]; then
        cp "$src/mpy-cross/build/mpy-cross" "$TOOLS/mpy-cross-$tag"
    else
        cp "$src/mpy-cross/mpy-cross" "$TOOLS/mpy-cross-$tag"
    fi
    make -C "$src/ports/unix" submodules "${flags[@]}" >/dev/null
    local runtime_cflags="$cflags -DMICROPY_FLOAT_IMPL=MICROPY_FLOAT_IMPL_FLOAT"

    if [ "$PLATFORM" = darwin ]; then
        local arch
        for arch in arm64 x86_64; do
            make -C "$src/ports/unix" BUILD="build-$arch" clean >/dev/null
            make -C "$src/ports/unix" -j8 BUILD="build-$arch" "${flags[@]}" \
                CFLAGS_EXTRA="$runtime_cflags" CC="clang -arch $arch" >/dev/null
            unix_binary "$src" "build-$arch" "$TOOLS/micropython-$tag-$arch"
        done
        lipo -create -output "$OUT/micropython-$tag" \
            "$TOOLS/micropython-$tag-arm64" "$TOOLS/micropython-$tag-x86_64"
    else
        make -C "$src/ports/unix" BUILD=build-linux clean >/dev/null
        make -C "$src/ports/unix" -j8 BUILD=build-linux "${flags[@]}" CFLAGS_EXTRA="$runtime_cflags" >/dev/null
        unix_binary "$src" build-linux "$OUT/micropython-$tag"
    fi
    cp "$src/LICENSE" "$OUT/LICENSE-micropython-$tag"
    echo "$OUT/micropython-$tag"
}

grep -v '^#' "$PIN_FILE" | while read -r _dialect tag commit; do
    if [ -n "$tag" ]; then
        build_one "$tag" "$commit"
    fi
done
