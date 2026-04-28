#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Firmware management tools — config backup, INI push, homing validation.

All tools use the production driver stack (SerialBoard/MotorBoard/LEDBoard).
For firmware flashing (UF2), use drivers/firmware_updater.py directly.

Usage:
    python -m tools.firmware_tools backup              # backup all config files
    python -m tools.firmware_tools push-ini             # push latest INI files to board
    python -m tools.firmware_tools homing-test          # 50-cycle homing endurance
    python -m tools.firmware_tools homing-test --cycles 100 --axes Z T
    python -m tools.firmware_tools info                 # show board info
    python -m tools.firmware_tools deploy --board motor --firmware path/to/main.py
    python -m tools.firmware_tools deploy --board led   --firmware path/to/main.py
    python -m tools.firmware_tools factory-reset --nuke-uf2 flash_nuke.uf2 \\
         --runtime-uf2 motor_runtime.uf2 --main-py path/to/main.py
    python -m tools.firmware_tools restore-configs --board motor \\
         --backup-dir path/to/backup
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add LumaViewPro root to path so imports work when run as script
_LVP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LVP_ROOT))

from drivers.motorboard import MotorBoard
from drivers.serialboard import SerialBoard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect_motor_board():
    """Connect to motor board using production MotorBoard driver."""
    board = MotorBoard()
    if not board.found:
        print('ERROR: Motor board not found (VID=0x2E8A, PID=0x0005)')
        sys.exit(1)
    board.connect()
    if not board.is_connected():
        print('ERROR: Failed to connect to motor board')
        sys.exit(1)
    return board


def _connect_serial_board(vid=0x2E8A, pid=0x0005, label='[Tool]',
                          timeout=5, write_timeout=5):
    """Connect a raw SerialBoard (for raw REPL operations without MotorBoard overhead)."""
    board = SerialBoard(vid, pid, label, timeout=timeout,
                        write_timeout=write_timeout)
    if not board.found:
        print(f'ERROR: Board not found (VID=0x{vid:04X}, PID=0x{pid:04X})')
        sys.exit(1)
    board.connect()
    if not board.is_connected():
        print('ERROR: Failed to connect')
        sys.exit(1)
    return board


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args):
    """Show board info via FULLINFO."""
    board = _connect_motor_board()
    try:
        resp = board.exchange_command('FULLINFO', response_numlines=1)
        if resp:
            print(resp)
        else:
            print('ERROR: No response from FULLINFO')
    finally:
        board.disconnect()


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------

def cmd_backup(args):
    """Backup all config files from motor board via raw REPL."""
    board = _connect_serial_board(label='[Backup]')

    # Create backup directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(args.output) if args.output else Path(f'build/config_backup_{timestamp}')
    backup_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f'Entering raw REPL...')
        if not board.enter_raw_repl():
            print('ERROR: Failed to enter raw REPL')
            sys.exit(1)

        # List files
        files = board.repl_list_files()
        print(f'Files on board: {files}')

        # Backup each file
        config_files = ['motorconfig.json', 'xymotorconfig.ini',
                        'ztmotorconfig.ini', 'ztmotorconfig2.ini', 'main.py']
        backed_up = []
        for filename in config_files:
            if filename not in files:
                print(f'  {filename}: not present, skipping')
                continue
            data = board.repl_read_file(filename)
            if data is None:
                print(f'  {filename}: read failed')
                continue
            out_path = backup_dir / filename
            out_path.write_bytes(data)
            print(f'  {filename}: {len(data)} bytes -> {out_path}')
            backed_up.append(filename)

        print(f'\nBackup complete: {len(backed_up)} files -> {backup_dir}')

    finally:
        board.exit_raw_repl()
        resp = board.verify_firmware_running()
        if resp:
            print(f'Firmware running: {resp[:60]}')
        board.disconnect()


# ---------------------------------------------------------------------------
# push-ini
# ---------------------------------------------------------------------------

def cmd_push_ini(args):
    """Push latest INI files from data/firmware/ to motor board."""
    ini_dir = _LVP_ROOT / 'data' / 'firmware'
    ini_files = ['xymotorconfig.ini', 'ztmotorconfig.ini', 'ztmotorconfig2.ini']

    if args.files:
        ini_files = args.files

    # Verify source files exist
    for name in ini_files:
        src = ini_dir / name
        if not src.exists():
            print(f'ERROR: Source file not found: {src}')
            sys.exit(1)

    board = _connect_serial_board(label='[INI Push]')

    try:
        print('Entering raw REPL...')
        if not board.enter_raw_repl():
            print('ERROR: Failed to enter raw REPL')
            sys.exit(1)

        for name in ini_files:
            src = ini_dir / name
            data = src.read_bytes()
            print(f'Writing {name} ({len(data)} bytes)...')
            ok = board.repl_write_file(name, data)
            if ok:
                print(f'  {name}: OK (SHA256 verified)')
            else:
                print(f'  {name}: FAILED')

        # Verify files are readable
        print('\nVerifying...')
        for name in ini_files:
            readback = board.repl_read_file(name)
            src_data = (ini_dir / name).read_bytes()
            if readback == src_data:
                print(f'  {name}: verified')
            else:
                print(f'  {name}: MISMATCH')

    finally:
        board.exit_raw_repl()
        resp = board.verify_firmware_running()
        if resp:
            print(f'Firmware running: {resp[:60]}')
        else:
            print('WARNING: Firmware not responding after raw REPL exit')
        board.disconnect()


# ---------------------------------------------------------------------------
# homing-test
# ---------------------------------------------------------------------------

def _get_position_steps(board, axis):
    """Read raw step position for an axis."""
    return board.current_pos_steps(axis)


def _wait_for_stop(board, axis, timeout=30):
    """Wait until axis reports position_reached (bit 9 of STATUS)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        resp = board.exchange_command(f'STATUS_R{axis}')
        if resp is None:
            time.sleep(0.1)
            continue
        try:
            status = int(resp)
        except ValueError:
            time.sleep(0.1)
            continue
        if status & 0x200:  # bit 9 = position_reached
            return True
        time.sleep(0.1)
    return False


def _move_to_step(board, axis, position, timeout=30):
    """Move axis to absolute step position and wait."""
    board.exchange_command(f'TARGET_W{axis}{position}')
    return _wait_for_stop(board, axis, timeout)


def _home_single(board, axis, timeout=30):
    """Home a single axis. Returns (success, response, duration_ms)."""
    cmd = f'{axis}HOME'
    t0 = time.monotonic()
    resp = board.exchange_command(cmd)
    dt = (time.monotonic() - t0) * 1000
    if resp is None:
        return False, 'No response', dt
    ok = ('successful' in resp.lower() or 'complete' in resp.lower())
    return ok, resp.strip(), dt


def _home_all(board, timeout=300):
    """HOME command (homes all axes). Returns (success, response, duration_ms)."""
    # HOME can take 60s+ for all axes. MotorBoard has 30s timeout which is
    # sufficient since the firmware sends the response when done.
    t0 = time.monotonic()
    resp = board.exchange_command('HOME')
    dt = (time.monotonic() - t0) * 1000
    if resp is None:
        return False, 'No response (timeout?)', dt
    ok = ('successful' in resp.lower() or 'complete' in resp.lower()
          or 'not present' in resp.lower())
    return ok, resp.strip(), dt


def cmd_homing_test(args):
    """Run homing endurance test — N cycles, check position repeatability."""
    # Per-axis tolerances (steps). XY Hall effect sensors have inherent
    # hysteresis; Z/T optical interrupters are very repeatable.
    TOLERANCE = {
        'X': 10000,  # ~0.5mm — XY Hall sensor hysteresis
        'Y': 10000,
        'Z': 50,     # very repeatable (optical, slow double-pass)
        'T': 200,    # turret detent repeatability
    }

    board = _connect_motor_board()

    try:
        # Detect axes
        present = board.detect_present_axes()
        print(f'Present axes: {present}')

        if args.axes:
            axes = [a.upper() for a in args.axes]
            missing = [a for a in axes if a not in present]
            if missing:
                print(f'WARNING: Requested axes {missing} not present, skipping')
                axes = [a for a in axes if a in present]
        else:
            axes = present

        if not axes:
            print('ERROR: No axes to test.')
            sys.exit(1)

        n_cycles = args.cycles
        move_between = args.move_between

        # Initial home to establish reference
        print(f'\n--- Initial home (establishing reference) ---')
        if len(axes) == 1:
            ok, resp, dt = _home_single(board, axes[0])
        else:
            ok, resp, dt = _home_all(board)

        if not ok:
            print(f'FAIL: Initial home failed: {resp}')
            sys.exit(1)

        ref_positions = {}
        for axis in axes:
            pos = _get_position_steps(board, axis)
            if pos is None:
                print(f'ERROR: Cannot read {axis} position')
                sys.exit(1)
            ref_positions[axis] = pos

        print(f'Reference positions: {ref_positions}')
        print(f'Initial home took {dt:.0f}ms')

        # Move-away targets
        move_targets = {}
        for axis in axes:
            if axis == 'Z':
                move_targets[axis] = ref_positions[axis] + 3000
            elif axis == 'T':
                move_targets[axis] = ref_positions[axis] + 2000
            else:  # X, Y
                move_targets[axis] = ref_positions[axis] + 5000

        print(f'\nStarting {n_cycles} homing cycles on axes {axes}')
        print(f'Move between cycles: {move_between}')
        if move_between:
            print(f'Move-away targets: {move_targets}')
        print(f'{"="*70}')

        results = []
        for cycle in range(1, n_cycles + 1):
            cycle_result = {
                'cycle': cycle,
                'success': True,
                'errors': [],
                'positions': {},
                'position_deltas': {},
                'home_time_ms': 0,
            }

            # Move away from home if requested
            if move_between:
                for axis in axes:
                    arrived = _move_to_step(board, axis, move_targets[axis], timeout=20)
                    if not arrived:
                        cycle_result['errors'].append(f'{axis} move-away timeout')
                time.sleep(0.2)

            # Home
            if len(axes) == 1:
                ok, resp, dt = _home_single(board, axes[0], timeout=30)
            else:
                ok, resp, dt = _home_all(board, timeout=60)

            cycle_result['home_time_ms'] = dt

            if not ok:
                cycle_result['success'] = False
                cycle_result['errors'].append(f'Home failed: {resp}')
            else:
                for axis in axes:
                    pos = _get_position_steps(board, axis)
                    if pos is None:
                        cycle_result['success'] = False
                        cycle_result['errors'].append(f'{axis} position read failed')
                        continue
                    cycle_result['positions'][axis] = pos
                    delta = pos - ref_positions[axis]
                    cycle_result['position_deltas'][axis] = delta
                    tol = TOLERANCE.get(axis, 200)
                    if abs(delta) > tol:
                        cycle_result['success'] = False
                        cycle_result['errors'].append(
                            f'{axis} position drift: {delta} steps '
                            f'(ref={ref_positions[axis]}, actual={pos})')

            results.append(cycle_result)

            # Print status
            status = 'OK' if cycle_result['success'] else 'FAIL'
            deltas = ' '.join(
                f'{a}={cycle_result["position_deltas"].get(a, "?")}'
                for a in axes)
            errors = '; '.join(cycle_result['errors']) if cycle_result['errors'] else ''
            extra = f'  ** {errors}' if errors else ''
            print(f'[{cycle:3d}/{n_cycles}] {status}  {dt:6.0f}ms  deltas: {deltas}{extra}')

        # Print summary
        _print_homing_summary(results, axes)

    finally:
        board.disconnect()


def _print_homing_summary(results, axes):
    """Print homing endurance test summary."""
    n = len(results)
    if n == 0:
        print('No results.')
        return

    passed = sum(1 for r in results if r['success'])
    failed = n - passed
    times = [r['home_time_ms'] for r in results]

    print(f'\n{"="*70}')
    print(f'HOMING ENDURANCE TEST SUMMARY')
    print(f'{"="*70}')
    print(f'Cycles: {n}')
    print(f'Passed: {passed}')
    print(f'Failed: {failed}')
    print(f'Pass rate: {100*passed/n:.1f}%')
    print()
    print(f'Homing time (ms):')
    print(f'  Min:  {min(times):.0f}')
    print(f'  Max:  {max(times):.0f}')
    print(f'  Mean: {sum(times)/len(times):.0f}')
    print()

    for axis in axes:
        deltas = [r['position_deltas'].get(axis, 0) for r in results if r['success']]
        if deltas:
            print(f'{axis} position delta (steps from reference):')
            print(f'  Min:  {min(deltas)}')
            print(f'  Max:  {max(deltas)}')
            print(f'  Mean: {sum(deltas)/len(deltas):.1f}')
        else:
            print(f'{axis}: no successful cycles')

    if failed:
        print(f'\nFAILURES:')
        for r in results:
            if not r['success']:
                print(f'  Cycle {r["cycle"]}: {"; ".join(r["errors"])}')

    print(f'{"="*70}')
    return failed == 0


# ---------------------------------------------------------------------------
# deploy — firmware update via Lumascope API (Phase 4E)
# ---------------------------------------------------------------------------

def cmd_deploy(args):
    """Deploy firmware via Lumascope.update_*_firmware API methods.

    Routes through modules.lumascope_api so GUI + CLI + automated tests
    share the same entry point. Does not reach into drivers.firmware_
    updater directly.
    """
    from modules.lumascope_api import Lumascope

    method = args.method
    board = args.board
    fw_path = Path(args.firmware)

    if not fw_path.is_file():
        print(f'ERROR: firmware file not found: {fw_path}')
        sys.exit(1)

    if board == 'led' and method == 'uf2':
        print('ERROR: LED board has no direct USB — UF2 path is not '
              'supported. Use --method repl.')
        sys.exit(1)

    def _progress(stage, msg, fraction):
        pct = int(fraction * 100)
        print(f'  [{pct:3d}%] {stage.value}: {msg}')

    print('Constructing diagnostic Lumascope (LED + motor, no camera)...')
    # create_diagnostic is the no-camera __new__ path used by tech-support
    # tooling; wires the full motion-monitor + Phase 4D event callbacks
    # through the same code as __init__ without paying the camera init
    # cost.
    scope = Lumascope.create_diagnostic()

    print(f'Deploying firmware to {board} board via {method} '
          f'({fw_path.stat().st_size} bytes from {fw_path})...')

    if board == 'motor':
        if method == 'repl':
            result = scope.update_motor_firmware(
                str(fw_path), progress_callback=_progress,
                skip_config_backup=args.skip_config_backup,
                skip_post_test=args.skip_post_test,
            )
        else:  # uf2
            result = scope.update_motor_firmware_uf2(
                str(fw_path), progress_callback=_progress,
                skip_config_backup=args.skip_config_backup,
                skip_post_test=args.skip_post_test,
            )
    else:  # led
        result = scope.update_led_firmware(
            str(fw_path), progress_callback=_progress,
            skip_config_backup=args.skip_config_backup,
            skip_post_test=args.skip_post_test,
        )

    print()
    print('=== UpdateResult ===')
    print(f'  success:       {result.success}')
    print(f'  board_type:    {result.board_type}')
    print(f'  old_version:   {result.old_version}')
    print(f'  new_version:   {result.new_version}')
    print(f'  backup_path:   {result.config_backup_path}')
    if result.error_stage is not None:
        print(f'  error_stage:   {result.error_stage}')
    if result.error_message:
        print(f'  error_message: {result.error_message}')
    if result.warnings:
        print(f'  warnings:')
        for w in result.warnings:
            print(f'    - {w}')

    sys.exit(0 if result.success else 2)


# ---------------------------------------------------------------------------
# upgrade — FW4.0 field-upgrade via Lumascope API (FIRMWARE_PLAN §13.X)
# ---------------------------------------------------------------------------

def cmd_upgrade(args):
    """Field-upgrade a board to FW4.0 via the bundled source tree.

    Secondary interface to Lumascope.upgrade_board_fw40 (engineering,
    factory, support debugging). LVP is the primary caller — §13.X.
    Exit codes disjoint from other subcommands:
        0   success
        10  P0 host source validation failed
        20  P1 probe classified board as unresponsive
        30  P2 config backup failed
        35  P2 Overwritable flag blocked the requested write
        40  P4 bundle write failed
        50  P5 post-flash verify failed
        2   CLI argument error
    """
    from modules.lumascope_api import Lumascope

    source_tree = Path(args.source)
    if not source_tree.is_dir():
        print(f'ERROR: --source path is not a directory: {source_tree}')
        sys.exit(2)

    def _progress(stage, msg, fraction):
        pct = int(fraction * 100)
        print(f'  [{pct:3d}%] {stage.value}: {msg}')

    print('Constructing diagnostic Lumascope (LED + motor, no camera)...')
    scope = Lumascope.create_diagnostic()

    print(f'Upgrading {args.board} board from {source_tree} ...')
    if args.dry_run:
        print('  (--dry-run: P0 host validation only, no transport)')
    if args.ignore_overwritable:
        print('  (--ignore-overwritable: Overwritable flags bypassed)')

    result = scope.upgrade_board_fw40(
        args.board, source_tree,
        dry_run=args.dry_run,
        respect_overwritable=not args.ignore_overwritable,
        progress_callback=_progress,
    )

    print()
    print('=== UpgradeResult ===')
    print(f'  success:          {result.success}')
    print(f'  exit_code:        {result.exit_code}')
    print(f'  board_type:       {result.board_type}')
    print(f'  old_version:      {result.old_version}')
    print(f'  new_version:      {result.new_version}')
    print(f'  probe:            {result.probe_classification}')
    print(f'  backup_path:      {result.config_backup_path}')
    print(f'  telemetry:        {result.telemetry_log_path}')
    if result.overwritable_flags:
        print(f'  overwritable:     {result.overwritable_flags}')
    if result.files_written:
        print(f'  files_written:    {result.files_written}')
    if result.files_skipped_overwritable:
        print(f'  skipped (I5):     {result.files_skipped_overwritable}')
    if result.error_code:
        print(f'  error_code:       {result.error_code}')
    if result.error_message:
        print(f'  error_message:    {result.error_message}')
    if result.error_stage is not None:
        print(f'  error_stage:      {result.error_stage}')
    if result.warnings:
        print('  warnings:')
        for w in result.warnings:
            print(f'    - {w}')

    sys.exit(result.exit_code)


# ---------------------------------------------------------------------------
# factory-reset — motor-only full recovery via Lumascope API (Phase 4F)
# ---------------------------------------------------------------------------

def cmd_factory_reset(args):
    """Factory-reset a motor board whose firmware blocks raw REPL.

    Chains nuke -> runtime UF2 flash -> main.py push via the
    Lumascope.factory_reset_motor API method.
    """
    from modules.lumascope_api import Lumascope

    nuke_uf2 = Path(args.nuke_uf2)
    runtime_uf2 = Path(args.runtime_uf2)
    main_py = Path(args.main_py)

    for p, label in ((nuke_uf2, 'nuke UF2'),
                     (runtime_uf2, 'runtime UF2'),
                     (main_py, 'main.py')):
        if not p.is_file():
            print(f'ERROR: {label} not found: {p}')
            sys.exit(1)

    def _progress(stage, msg, fraction):
        pct = int(fraction * 100)
        print(f'  [{pct:3d}%] {stage.value}: {msg}')

    print('Constructing diagnostic Lumascope (LED + motor, no camera)...')
    scope = Lumascope.create_diagnostic()

    print(f'Factory resetting motor board:')
    print(f'  nuke UF2:    {nuke_uf2}')
    print(f'  runtime UF2: {runtime_uf2}')
    print(f'  main.py:     {main_py}')
    print()

    result = scope.factory_reset_motor(
        str(nuke_uf2), str(runtime_uf2), str(main_py),
        progress_callback=_progress,
        skip_post_test=args.skip_post_test,
    )

    print()
    print('=== UpdateResult ===')
    print(f'  success:       {result.success}')
    print(f'  old_version:   {result.old_version}')
    print(f'  new_version:   {result.new_version}')
    if result.error_stage is not None:
        print(f'  error_stage:   {result.error_stage}')
    if result.error_message:
        print(f'  error_message: {result.error_message}')
    if result.warnings:
        print(f'  warnings:')
        for w in result.warnings:
            print(f'    - {w}')

    sys.exit(0 if result.success else 2)


# ---------------------------------------------------------------------------
# bench — driver-method round-trip latency measurement (release gate §2.3)
# ---------------------------------------------------------------------------
#
# Benchmarks driver-level methods (e.g. MotorBoard.fullinfo, LEDBoard.get_info)
# rather than raw firmware command strings. The driver dispatches v3.0.x vs
# FW4.0 internally, so running the bench against v3.0.x firmware then against
# FW4.0 firmware on the SAME hardware gives a directly comparable latency
# delta — the core §2.3 "≥20 ms improvement" evidence.
#
# The raw-command path remains available via `--raw-commands X,Y` for ad-hoc
# exploratory measurement (e.g. "is CALIBRATE slow?"). That mode requires
# the caller to know which command strings exist on the firmware running.


def _board_bench_callables(board_kind, board):
    """Default driver-method bench set per board kind.

    Uses each board's `_connect_bench_callables()` so the CLI and the
    connect-time fingerprint benchmark the exact same set — one source
    of truth, no drift. A board that overrides the hook to add a
    second method is covered everywhere automatically.
    """
    getter = getattr(board, '_connect_bench_callables', None)
    if getter is None:
        return []
    return list(getter())


def _write_bench_csv(path, rows):
    """Write per-iteration rows to CSV.

    rows: iterable of (board, firmware_version, method, iteration,
    duration_us). duration_us is None for failed iterations.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['board', 'firmware_version', 'method',
                         'iteration', 'duration_us'])
        for row in rows:
            writer.writerow(row)


def _format_summary_table(per_method_summary):
    """Format per-method summary dict as a plain-text aligned table."""
    hdr = ('method', 'count', 'err', 'mean_us', 'stddev',
           'p50', 'p95', 'p99', 'min', 'max')
    widths = [max(len(hdr[i]), 8) for i in range(len(hdr))]
    for name in per_method_summary:
        widths[0] = max(widths[0], len(name))
    lines = ['  '.join(h.ljust(widths[i]) for i, h in enumerate(hdr))]
    lines.append('  '.join('-' * widths[i] for i in range(len(hdr))))
    for name, s in per_method_summary.items():
        def fmt(v):
            return '{:.1f}'.format(v) if isinstance(v, float) else str(v) if v is not None else '—'
        row = (name, s['count'], s['errors'], fmt(s['mean_us']),
               fmt(s['stddev_us']), fmt(s['p50_us']), fmt(s['p95_us']),
               fmt(s['p99_us']), fmt(s['min_us']), fmt(s['max_us']))
        lines.append('  '.join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
    return '\n'.join(lines)


def cmd_bench(args):
    """Measure round-trip latency of driver methods on the connected board.

    Validates the release gate §2.3 protocol-latency thesis. Run once
    against v3.0.x, once against FW4.0, on the SAME hardware, to compare.

    Hardware-gated: requires a connected, responsive board.
    """
    from drivers import serial_latency
    from modules.lumascope_api import Lumascope

    output_path = Path(args.output) if args.output else None
    iterations = args.iterations
    warmup = args.warmup

    print('Constructing diagnostic Lumascope (LED + motor, no camera)...')
    scope = Lumascope.create_diagnostic()

    board = scope.motion if args.board == 'motor' else scope.led

    if not hasattr(board, 'is_connected') or not board.is_connected():
        print(f'ERROR: {args.board} board not connected / not responding')
        scope.disconnect()
        sys.exit(1)

    # Build the callable set.
    if args.raw_commands:
        raw_cmds = tuple(c.strip() for c in args.raw_commands.split(',') if c.strip())
        named = [(cmd, (lambda c=cmd: board.exchange_command(c))) for cmd in raw_cmds]
        mode_label = f'raw commands [{", ".join(raw_cmds)}]'
    else:
        named = _board_bench_callables(args.board, board)
        mode_label = f'driver methods [{", ".join(n for n, _ in named)}]'

    if not named:
        print(f'ERROR: no bench methods for {args.board} board')
        scope.disconnect()
        sys.exit(1)

    fw_version = getattr(board, 'firmware_version', None) or 'unknown'
    print(f'Benchmarking {args.board} board at firmware {fw_version}')
    print(f'Mode:     {mode_label}')
    print(f'Warmup:   {warmup} iterations (discarded)')
    print(f'Measured: {iterations} iterations per method')
    if output_path:
        print(f'CSV output: {output_path}')
    print()

    all_rows = []
    per_method = {}

    try:
        # One measurement pass: primitive returns both summaries and
        # raw per-iteration durations when `return_durations=True`.
        per_method, raw_by_name = serial_latency.measure_callable_latencies(
            named, iterations=iterations, warmup=warmup,
            return_durations=True,
        )
        for name, summary in per_method.items():
            if summary['count'] > 0:
                print(f'  {name}: mean={summary["mean_us"]:.1f}µs '
                      f'p95={summary["p95_us"]:.1f}µs '
                      f'errors={summary["errors"]}')
            else:
                print(f'  {name}: ALL FAILED ({summary["errors"]} errors)')
            if output_path:
                for i, d in enumerate(raw_by_name[name]):
                    all_rows.append((args.board, fw_version, name, i, d))
    finally:
        scope.disconnect()

    print()
    print('=== Summary (µs) ===')
    print(_format_summary_table(per_method))

    if output_path:
        _write_bench_csv(output_path, all_rows)
        print(f'\nPer-iteration data written to {output_path}')

    # Optional reliability loop (release gate §2.3 "Reliability under load").
    if args.load_minutes is not None and args.load_minutes > 0:
        duration_s = args.load_minutes * 60.0
        print()
        print(f'=== Reliability loop ===')
        # Pick the first benched callable for the load target. Override
        # via --raw-commands if you want to hammer a different command.
        load_name, load_fn = named[0]
        print(f'Target:   {load_name}')
        print(f'Rate:     {args.load_hz:.1f} Hz target')
        print(f'Duration: {args.load_minutes:.1f} min ({duration_s:.1f} s)')
        print('  running... (Ctrl-C to abort)')

        scope2 = Lumascope.create_diagnostic()
        try:
            board2 = scope2.motion if args.board == 'motor' else scope2.led
            # Rebind the callable against the fresh scope's board.
            named2 = _board_bench_callables(args.board, board2)
            name_to_fn = dict(named2)
            load_fn2 = name_to_fn.get(load_name, load_fn)
            load_summary = serial_latency.run_load_loop(
                load_fn2, duration_s, args.load_hz
            )
        finally:
            scope2.disconnect()

        print()
        print(f'  count:            {load_summary["count"]}')
        print(f'  errors:           {load_summary["errors"]}')
        print(f'  errors/hour:      {load_summary["errors_per_hour"]:.1f}')
        print(f'  duration (s):     {load_summary["duration_s"]:.2f}')
        print(f'  target_hz:        {load_summary["target_hz"]:.2f}')
        print(f'  actual_hz:        {load_summary["actual_hz"]:.2f}')
        if load_summary['count'] > 0:
            print(f'  mean (µs):        {load_summary["mean_us"]:.1f}')
            print(f'  p50/p95/p99 (µs): {load_summary["p50_us"]:.1f} / '
                  f'{load_summary["p95_us"]:.1f} / {load_summary["p99_us"]:.1f}')


# ---------------------------------------------------------------------------
# reliability-soak — host↔board comm health gate (tech-support / QC bringup)
# ---------------------------------------------------------------------------

def cmd_reliability_soak(args):
    """Run an alternating-LED-command soak + RTT histogram against the
    LED board and report PASS/FAIL.

    Intended uses:
      - QC mainboard bringup gate before signing off a board.
      - tech-support report supplement when a customer reports flaky LED
        behavior — gives objective host↔board comm-layer health vs.
        application-layer hypotheses.
      - regression check after any driver / firmware change that touches
        the serial framing path.

    Tests run:
      1. Pair soak: N × (LED_SET ch=0 50mA + LED_OFF ch=0). Detects
         framing desync, partial-line reads, late-byte drops.
      2. (optional) Multi-channel cycle: M × ch 0-5 LED_SET/LED_OFF.
         Stresses dispatcher under broader command surface.
      3. Per-command RTT histogram: 200 samples each of LED_SET,
         LED_OFF, STATUS, HEAP, LED_READ <ch>, INFO.
      4. Heap delta probe: HEAP free before/after the soak. Detects
         per-command memory leak.

    PASS criteria (default):
      - errors == 0 across all command pairs
      - heap_delta < 1024 bytes (no leak)
      - p99 RTT(LED_SET) < 100 ms (framing healthy)

    Exits 0 on PASS, 1 on FAIL. Detailed report printed to stdout.
    """
    import time
    import statistics
    from drivers.ledboard import LEDBoard

    print('[reliability-soak] Connecting to LED board...')
    try:
        board = LEDBoard()
    except Exception as e:
        print(f'[reliability-soak] Connect failed: {e}')
        sys.exit(1)

    is_v35 = board._use_v35()
    proto = board.protocol_version.value
    print(f'[reliability-soak] protocol={proto} '
          f'fw={board.firmware_version} '
          f'features={len(board.features)}')

    if not board.firmware_responding:
        print('[reliability-soak] FAIL: board not responding to INFO')
        sys.exit(1)

    issues = []

    # Protocol-agnostic framing-health check. v3.5 returns 'OK', legacy
    # returns the command echo; both should be free of error tokens.
    def _ok(r):
        if r is None:
            return False
        s = str(r).upper()
        return not any(t in s for t in (
            'ERROR', 'INVALID', 'COMMAND NOT RECOGNIZED', 'FAIL'))

    def heap():
        # HEAP is v3.5-only. Legacy LED firmware has no HEAP command, so
        # the leak probe is unavailable on configurations A and B.
        if not is_v35:
            return None
        r = board.exchange_command('HEAP', timeout=2)
        if not r:
            return None
        for tok in r.split():
            if tok.startswith('free='):
                try:
                    return int(tok.split('=', 1)[1])
                except ValueError:
                    return None
        return None

    heap_pre = heap()
    if heap_pre is not None:
        print(f'[reliability-soak] heap_pre={heap_pre}')
    else:
        print('[reliability-soak] heap probe: skipped (legacy firmware)')

    # Build per-protocol wire strings via the driver's helpers — keeps
    # all wire-spelling decisions in the driver layer. Without this the
    # tool hardcoded v3.5 strings ('LED_SET 0 50' etc.), which legacy
    # firmware parses as channel '_' and rejects, producing 100% errors
    # on configurations A/B (audit F1, 2026-04-27).
    _, set_cmd_ch0 = board._build_led_on_cmd(channel=0, mA=50)
    _, off_cmd_ch0 = board._build_led_off_cmd(channel=0)

    # Ensure channels are enabled.
    enable_cmd = 'LED_ENABLE ALL' if is_v35 else 'LEDS_ENT'
    board.exchange_command(enable_cmd, timeout=2)

    # ---- Test 1: pair soak ----
    n = args.iters
    print(f'[reliability-soak] Test 1: {n}× LED_SET/LED_OFF ch=0 (~{n*0.04:.0f}s)')
    err_t1 = 0
    slow_t1 = 0
    times_t1_api = []   # outer wall-clock (includes lock + drain + log)
    times_t1_wire = []  # inner wire-only (write→last-readline)
    t_start = time.monotonic()
    for i in range(n):
        t = time.monotonic()
        r1, w1 = board.exchange_command(set_cmd_ch0, return_timing=True)
        r2, w2 = board.exchange_command(off_cmd_ch0, return_timing=True)
        dt = (time.monotonic() - t) * 1000
        times_t1_api.append(dt)
        wire_pair_ms = ((w1 or 0.0) + (w2 or 0.0)) * 1000
        times_t1_wire.append(wire_pair_ms)
        if not _ok(r1) or not _ok(r2):
            err_t1 += 1
        if dt > args.slow_threshold_ms:
            slow_t1 += 1
    elapsed_t1 = time.monotonic() - t_start
    times_t1_api.sort()
    times_t1_wire.sort()
    print(f'  done {elapsed_t1:.1f}s err={err_t1}/{n} '
          f'slow(>{args.slow_threshold_ms}ms)={slow_t1}')
    print(f'  pair API   p50={times_t1_api[n//2]:6.2f} mean={statistics.mean(times_t1_api):6.2f} '
          f'p99={times_t1_api[int(n*0.99)]:6.2f} max={times_t1_api[-1]:6.2f} ms')
    print(f'  pair WIRE  p50={times_t1_wire[n//2]:6.2f} mean={statistics.mean(times_t1_wire):6.2f} '
          f'p99={times_t1_wire[int(n*0.99)]:6.2f} max={times_t1_wire[-1]:6.2f} ms')
    if err_t1 > 0:
        issues.append(f'Test 1 errors: {err_t1}/{n}')

    # ---- Test 2: multi-channel cycle (optional) ----
    if args.multi_channel:
        m = args.multi_iters
        print(f'[reliability-soak] Test 2: {m}× cycle ch 0-5 ({m*12} cmds)')
        err_t2 = 0
        t_start = time.monotonic()
        for _ in range(m):
            for ch in range(6):
                _, set_cmd = board._build_led_on_cmd(channel=ch, mA=50)
                _, off_cmd = board._build_led_off_cmd(channel=ch)
                r1 = board.exchange_command(set_cmd)
                r2 = board.exchange_command(off_cmd)
                if not _ok(r1) or not _ok(r2):
                    err_t2 += 1
        elapsed_t2 = time.monotonic() - t_start
        print(f'  done {elapsed_t2:.1f}s err={err_t2}/{m*12}')
        if err_t2 > 0:
            issues.append(f'Test 2 errors: {err_t2}/{m*12}')

    # ---- Test 3: per-command RTT ----
    if args.rtt_histogram:
        print(f'[reliability-soak] Test 3: per-command RTT (200 samples each)')

        def bench(cmd, samples=200):
            s_api = []
            s_wire = []
            for _ in range(samples):
                t = time.monotonic()
                _, w = board.exchange_command(
                    cmd, timeout=2, return_timing=True)
                s_api.append((time.monotonic() - t) * 1000)
                s_wire.append((w or 0.0) * 1000)
            s_api.sort()
            s_wire.sort()
            return (s_api[samples // 2], statistics.mean(s_api), s_api[int(samples * 0.99)],
                    s_wire[samples // 2], statistics.mean(s_wire), s_wire[int(samples * 0.99)])

        # Histogram commands: v3.5 has STATUS/HEAP/LED_READ; legacy has
        # none of those (LEDREAD requires engineering mode and is not
        # entered here, audit F14). Restrict to commands the firmware
        # actually answers, by protocol.
        if is_v35:
            histo_cmds = [set_cmd_ch0, off_cmd_ch0,
                          'STATUS', 'HEAP', 'LED_READ 0', 'INFO']
        else:
            histo_cmds = [set_cmd_ch0, off_cmd_ch0, 'INFO']

        print(f'  {"CMD":<18s}   API_p50  API_mean  API_p99   WIRE_p50  WIRE_mean WIRE_p99')
        for cmd in histo_cmds:
            p50a, mna, p99a, p50w, mnw, p99w = bench(cmd)
            print(f'  {cmd:<18s}  {p50a:7.2f}  {mna:8.2f}  {p99a:7.2f}   '
                  f'{p50w:7.2f}  {mnw:8.2f}  {p99w:7.2f} ms')

    # ---- Test 4: heap delta ----
    board.exchange_command(board._build_leds_off_cmd())
    heap_post = heap()
    if heap_pre is not None and heap_post is not None:
        delta = heap_post - heap_pre
        print(f'[reliability-soak] heap_post={heap_post} delta={delta}')
        if delta < -args.heap_leak_threshold_bytes:
            issues.append(f'heap leak: {-delta} bytes lost')
    elif is_v35:
        print(f'[reliability-soak] heap probe failed pre={heap_pre} post={heap_post}')
    else:
        print('[reliability-soak] heap probe: skipped (legacy firmware)')

    # ---- Pass/fail criteria ----
    p99_t1 = times_t1_api[int(n * 0.99)]
    if p99_t1 > args.p99_threshold_ms:
        issues.append(f'pair RTT p99 too slow: {p99_t1:.1f}ms > {args.p99_threshold_ms}ms')

    print()
    print('=' * 60)
    if issues:
        print('RESULT: FAIL')
        for iss in issues:
            print(f'  - {iss}')
        print('=' * 60)
        sys.exit(1)
    else:
        print('RESULT: PASS')
        print('=' * 60)
        sys.exit(0)


# ---------------------------------------------------------------------------
# flash-dev-board — flash a UF2 to a bare RP2350 dev board (Pi Pico 2,
# Seeed XIAO RP2350, etc.). Uses drivers/firmware_updater.py:flash_dev_board()
# — production code path per Architecture Rule 22.
# ---------------------------------------------------------------------------

def cmd_flash_dev_board(args):
    """Flash a UF2 to a bare RP2350 dev board.

    Wraps drivers.firmware_updater.flash_dev_board(). Stdout shows progress
    + final status. Optional --probe runs a MicroPython script via raw REPL
    after flash and prints the captured output.
    """
    from pathlib import Path
    from drivers.firmware_updater import flash_dev_board, UpdateStage

    uf2_path = Path(args.uf2)
    if not uf2_path.is_file():
        print(f"ERROR: UF2 not found: {uf2_path}")
        sys.exit(1)

    probe_path = None
    if args.probe is not None:
        probe_path = Path(args.probe)
        if not probe_path.is_file():
            print(f"ERROR: probe script not found: {probe_path}")
            sys.exit(1)

    def progress(stage, message, fraction):
        bar_width = 30
        filled = int(bar_width * fraction)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  [{bar}] {stage.value:24s} {message}")

    print(f"=== flash-dev-board ===")
    print(f"  UF2:    {uf2_path}")
    print(f"  Port:   {args.port or '(auto-detect)'}")
    print(f"  Probe:  {probe_path or '(none)'}")
    print()

    result = flash_dev_board(
        uf2_path=uf2_path,
        port=args.port,
        progress_callback=progress,
        probe_script=probe_path,
        bootsel_timeout=args.bootsel_timeout,
    )

    print()
    if result.success:
        print("=== SUCCESS ===")
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        # If a probe script was run, the captured output is in the log at
        # INFO level. Re-print it here for the user.
        if probe_path is not None:
            print()
            print(f"Probe output ({probe_path.name}) is in the log; "
                  f"re-run with -v for visibility, or check stderr above.")
        sys.exit(0)
    else:
        print(f"=== FAILED ({result.error_stage}) ===")
        print(f"  {result.error_message}")
        for w in result.warnings:
            print(f"  warning: {w}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# restore-configs — push config backup to board via Lumascope API (Phase 4I)
# ---------------------------------------------------------------------------

def cmd_restore_configs(args):
    """Restore per-unit configs from a backup directory to a board.

    Symmetric counterpart of `backup` — writes motorconfig.json + INI
    files (motor) or cal.json (LED) from a local directory onto the
    board via raw REPL (SHA256-verified).
    """
    from modules.lumascope_api import Lumascope

    backup_dir = Path(args.backup_dir)
    if not backup_dir.is_dir():
        print(f'ERROR: backup directory not found: {backup_dir}')
        sys.exit(1)

    file_filter = set(args.files) if args.files else None

    def _progress(stage, msg, fraction):
        pct = int(fraction * 100)
        print(f'  [{pct:3d}%] {stage.value}: {msg}')

    print('Constructing diagnostic Lumascope (LED + motor, no camera)...')
    scope = Lumascope.create_diagnostic()

    print(f'Restoring configs to {args.board} board:')
    print(f'  backup dir: {backup_dir}')
    if file_filter:
        print(f'  filter:     {sorted(file_filter)}')
    print()

    if args.board == 'motor':
        result = scope.restore_motor_configs(
            str(backup_dir), progress_callback=_progress,
            file_filter=file_filter,
        )
    else:
        result = scope.restore_led_configs(
            str(backup_dir), progress_callback=_progress,
            file_filter=file_filter,
        )

    print()
    print('=== UpdateResult ===')
    print(f'  success:       {result.success}')
    print(f'  board_type:    {result.board_type}')
    print(f'  old_version:   {result.old_version}')
    if result.error_stage is not None:
        print(f'  error_stage:   {result.error_stage}')
    if result.error_message:
        print(f'  error_message: {result.error_message}')
    if result.warnings:
        print(f'  warnings:')
        for w in result.warnings:
            print(f'    - {w}')

    sys.exit(0 if result.success else 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Firmware management tools (uses production drivers)')
    sub = parser.add_subparsers(dest='command', help='Available commands')

    # info
    sub.add_parser('info', help='Show motor board info')

    # backup
    p_backup = sub.add_parser('backup', help='Backup config files from board')
    p_backup.add_argument('--output', '-o', default=None,
                          help='Output directory (default: build/config_backup_<timestamp>)')

    # push-ini
    p_ini = sub.add_parser('push-ini', help='Push INI files to motor board')
    p_ini.add_argument('--files', nargs='+', default=None,
                       help='Specific INI files to push (default: all 3)')

    # homing-test
    p_home = sub.add_parser('homing-test', help='Homing endurance test')
    p_home.add_argument('--cycles', type=int, default=50,
                        help='Number of homing cycles (default: 50)')
    p_home.add_argument('--axes', nargs='+', default=None,
                        help='Axes to test (default: all present)')
    p_home.add_argument('--move-between', action='store_true', default=True,
                        help='Move away from home between cycles (default: True)')
    p_home.add_argument('--no-move-between', action='store_false',
                        dest='move_between',
                        help='Skip intermediate moves')

    # deploy (Phase 4E)
    p_deploy = sub.add_parser('deploy', help='Deploy firmware via Lumascope API')
    p_deploy.add_argument('--board', choices=['motor', 'led'], required=True,
                          help='Target board')
    p_deploy.add_argument('--firmware', required=True,
                          help='Path to main.py (repl) or UF2 file (uf2)')
    p_deploy.add_argument('--method', choices=['repl', 'uf2'], default='repl',
                          help='Deploy method (default: repl). UF2 is motor-only.')
    p_deploy.add_argument('--skip-config-backup', action='store_true',
                          help='Skip motorconfig/cal backup')
    p_deploy.add_argument('--skip-post-test', action='store_true',
                          help='Skip post-update verification')

    # upgrade — FW4.0 field-upgrade (FIRMWARE_PLAN §13.X)
    p_upgrade = sub.add_parser(
        'upgrade',
        help=('Field-upgrade a board to FW4.0 from a Firmware-FW4.0 '
              'source tree. Exit codes: 0 success, 10 P0 source, '
              '20 P1 unresponsive, 30 P2 backup, 35 P2 Overwritable, '
              '40 P4 bundle, 50 P5 verify, 2 CLI error.'))
    p_upgrade.add_argument(
        '--board', choices=['motor', 'led'], required=True,
        help='Target board')
    p_upgrade.add_argument(
        '--source', required=True,
        help='Path to Firmware-FW4.0 repo root (contains '
             'firmware_manifest.json)')
    p_upgrade.add_argument(
        '--dry-run', action='store_true',
        help='Run P0 host source validation only; do not open transport.')
    p_upgrade.add_argument(
        '--ignore-overwritable', action='store_true',
        help='Bypass motorconfig.Overwritable flag checks. Proceeds '
             'with firmware write and logs a warning in telemetry. '
             'Engineering/factory escape hatch — do not use in field.')

    # factory-reset (Phase 4F) — motor-only recovery when raw REPL is broken
    p_reset = sub.add_parser(
        'factory-reset',
        help=('Full motor-board recovery: nuke -> runtime UF2 -> main.py push. '
              'Works from live firmware (sends FWUPDATE to enter BOOTSEL) OR from '
              'an already-in-BOOTSEL board (hold BOOTSEL and power-cycle first if '
              'firmware is wedged). Zero firmware-responsiveness assumed in the '
              'BOOTSEL entry path.'))
    p_reset.add_argument('--nuke-uf2', required=True,
                         help='Path to flash_nuke_rp2040.uf2 (wipes all flash)')
    p_reset.add_argument('--runtime-uf2', required=True,
                         help='Path to a clean MicroPython UF2 for the motor')
    p_reset.add_argument('--main-py', required=True,
                         help='Path to main.py to restore after reflash')
    p_reset.add_argument('--skip-post-test', action='store_true',
                         help='Skip post-update verification')

    # bench — driver-method round-trip latency (release gate §2.3)
    p_bench = sub.add_parser(
        'bench',
        help=('Measure driver-method round-trip latency on the connected '
              'board. Run against v3.0.x and FW4.0 on the same hardware '
              'to validate the release gate §2.3 protocol-latency thesis.'))
    p_bench.add_argument('--board', choices=['motor', 'led'], required=True,
                         help='Target board')
    p_bench.add_argument('--iterations', type=int, default=1000,
                         help='Measured iterations per method (default: 1000)')
    p_bench.add_argument('--warmup', type=int, default=50,
                         help='Warmup iterations per method, discarded '
                              '(default: 50)')
    p_bench.add_argument('--raw-commands', default=None,
                         help='Comma-separated raw firmware command list. '
                              'Escape hatch — bypasses the driver dispatcher '
                              'and sends each string via exchange_command(). '
                              'Use for ad-hoc measurement of a specific '
                              'firmware command string. Caller is responsible '
                              'for matching the firmware version running.')
    p_bench.add_argument('--output', default=None,
                         help='Optional CSV output path for per-iteration '
                              'durations')
    p_bench.add_argument('--load-minutes', type=float, default=None,
                         help='If set, runs a reliability loop after the '
                              'bench: targets the first benched method at '
                              '--load-hz for this many minutes. Release gate '
                              '§2.3 requires 10 Hz × 5 min to compare FW4.0 '
                              'vs v3.0.9 error rate under load.')
    p_bench.add_argument('--load-hz', type=float, default=10.0,
                         help='Target rate for the reliability loop '
                              '(default: 10 Hz per release gate §2.3).')

    # restore-configs (Phase 4I) — symmetric counterpart of `backup`
    p_restore = sub.add_parser(
        'restore-configs',
        help='Restore per-unit config backup to a board via raw REPL')
    p_restore.add_argument('--board', choices=['motor', 'led'], required=True,
                           help='Target board')
    p_restore.add_argument('--backup-dir', required=True,
                           help='Directory containing backed-up config files')
    p_restore.add_argument('--files', nargs='+', default=None,
                           help='Optional subset of filenames to restore '
                                '(default: all config files present in backup)')

    # flash-dev-board — flash a UF2 to a bare RP2350 dev board (Pi Pico 2,
    # Seeed XIAO RP2350, etc.). Uses drivers/firmware_updater.py:flash_dev_board()
    # — production code path per Architecture Rule 22, no parallel scripts.
    p_dev = sub.add_parser(
        'flash-dev-board',
        help=('Flash a UF2 to a bare RP2350 dev board (Pi Pico 2, Seeed XIAO '
              'RP2350, etc.). For board bring-up of un-firmwared dev boards. '
              'Uses picotool to push the running firmware into BOOTSEL if it '
              'supports the pico-sdk USB reset interface. Optionally runs a '
              'MicroPython probe script after flash via raw REPL.'))
    p_dev.add_argument('--uf2', required=True,
                       help='Path to UF2 file to flash')
    p_dev.add_argument('--port', default=None,
                       help='Optional explicit serial port. If omitted, '
                            'auto-discovers any known dev board via '
                            'drivers.firmware_updater.KNOWN_DEV_RP2350_PRE_FLASH')
    p_dev.add_argument('--probe', default=None,
                       help='Optional path to a MicroPython probe script to '
                            'exec on the device after flash (via raw REPL)')
    p_dev.add_argument('--bootsel-timeout', type=float, default=30.0,
                       help='Seconds to wait for BOOTSEL drive to mount '
                            '(default: 30)')

    # reliability-soak — host↔board comm health gate
    p_soak = sub.add_parser(
        'reliability-soak',
        help=('Run an alternating-LED-command soak + RTT histogram '
              'against the LED board and report PASS/FAIL. Intended '
              'for tech-support, QC bringup, regression checks.'))
    p_soak.add_argument(
        '--iters', type=int, default=1000,
        help='Pair-soak iterations (default: 1000)')
    p_soak.add_argument(
        '--multi-channel', action='store_true', default=True,
        help='Run multi-channel cycle test (default: True)')
    p_soak.add_argument(
        '--no-multi-channel', action='store_false', dest='multi_channel',
        help='Skip multi-channel cycle test')
    p_soak.add_argument(
        '--multi-iters', type=int, default=100,
        help='Multi-channel cycle iterations (default: 100; 12 cmds each)')
    p_soak.add_argument(
        '--rtt-histogram', action='store_true', default=True,
        help='Run per-command RTT histogram (default: True)')
    p_soak.add_argument(
        '--no-rtt-histogram', action='store_false', dest='rtt_histogram',
        help='Skip RTT histogram')
    p_soak.add_argument(
        '--slow-threshold-ms', type=int, default=200,
        help='Per-pair RTT to count as slow (default: 200ms)')
    p_soak.add_argument(
        '--p99-threshold-ms', type=int, default=100,
        help='Pair-RTT p99 PASS/FAIL threshold (default: 100ms)')
    p_soak.add_argument(
        '--heap-leak-threshold-bytes', type=int, default=1024,
        help='Heap shrink to count as leak (default: 1024B)')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        'info': cmd_info,
        'backup': cmd_backup,
        'push-ini': cmd_push_ini,
        'homing-test': cmd_homing_test,
        'deploy': cmd_deploy,
        'upgrade': cmd_upgrade,
        'factory-reset': cmd_factory_reset,
        'restore-configs': cmd_restore_configs,
        'bench': cmd_bench,
        'reliability-soak': cmd_reliability_soak,
        'flash-dev-board': cmd_flash_dev_board,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
