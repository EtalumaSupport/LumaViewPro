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
"""

import argparse
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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        'info': cmd_info,
        'backup': cmd_backup,
        'push-ini': cmd_push_ini,
        'homing-test': cmd_homing_test,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
