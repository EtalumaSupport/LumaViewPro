#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import re
import threading
import time
from lvp_logger import logger
from drivers.serialboard import SerialBoard, ProtocolVersion
from drivers.registry import led_registry


@led_registry.register('rp2040', priority=100)
class LEDBoard(SerialBoard):
    """LED controller (U50) driver — supports both protocol lanes:
    v3.0.x text (60 sealed field units, indefinite support) and v3.5
    short-text (new bench/factory units after the (A) decision). Per
    FIRMWARE_PROTOCOL.md §6 the two command sets share zero spellings;
    this class branches per detected protocol version.
    """

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(vid=0x0424, pid=0x704C, label='[LED Class ]')

        self._state_lock = threading.Lock()
        self.led_ma = {
            'BF': -1,
            'PC': -1,
            'DF': -1,
            'Red': -1,
            'Blue': -1,
            'Green': -1,
        }

        try:
            self.connect()
        except Exception:
            logger.error('[LED Class ] Failed to connect to LED controller')
            raise

        # Safety: immediately turn off all LEDs after connecting.
        # Old crashed LED firmware (pre-v3.0.4) can leave all LEDs stuck on
        # at full current (~500mA × 6 channels = 3A), causing thermal damage
        # to the board (measured 62°C). New v3.0.4+/v3.5 firmware initializes
        # LEDs off on boot, but this guard protects against old firmware
        # and interrupted previous sessions.
        self._safety_leds_off()

    def _accepts_json_info_fallback(self):
        """LED never speaks JSON on either v3.0.x or v3.5 — per the (A)
        decision the UART-bridge 16 B FIFO budget forces text framing on
        the LED controller forever (until RP2350 single-chip retires the
        bridge). Suppress the JSON-INFO fallback to skip the 1 s timeout
        on a wedged LED board, where the fallback would never recover."""
        return False

    # ------------------------------------------------------------------
    # v3.5 terminator-based exchange — reliability fix for the FIFO
    # bridge's occasional late-byte transmission.
    #
    # Bench data (2026-04-27, SN 7162-19): ~1-2% of alternating LED_SET /
    # LED_OFF iterations on v3.5 firmware showed framing desync — most
    # commonly an LED_OFF response read as a single 'R' character (host
    # readline timed out mid-transmission of 'RE: LED_OFF'). The 20 ms
    # post-response drain in serialboard.exchange_command is a
    # time-based heuristic; a 5 ms experiment showed the cushion isn't
    # the root cause, occasional late-arriving bytes are.
    #
    # Fix shape (Eric 2026-04-27): trust line CONTENT, not line count.
    # Read lines until the first non-empty non-RE: line (the spec §3.1
    # R2' terminator), then check in_waiting once with NO sleep — should
    # be 0 in healthy state, log+drain otherwise. Combined with
    # pyserial inter_byte_timeout=0.05 (host-side gap tolerance), this
    # makes the read path deterministic regardless of hub scheduling.
    # ------------------------------------------------------------------
    _V35_INTER_BYTE_TIMEOUT = 0.05  # seconds — readline returns when no
                                     # byte for this duration OR \n arrives.
                                     # Resets on each byte received, so
                                     # mid-transmission stalls don't break
                                     # frame.

    def _exchange_v35(self, command, timeout=2.0):
        """Send `command` and read until the first non-RE: terminator
        line. Per spec §3.1 R2' the firmware emits exactly one
        terminator per command (single-line self-terminating, or
        two-line ack with RE: + status, or multi-line streaming with
        END_<CMD> — for multi-line, callers should use exchange_multiline
        directly, not this method).

        Returns the terminator line as a string, or None on timeout /
        driver error.
        """
        with self._lock:
            if self.driver is None:
                try:
                    logger.info(f'{self._label} Auto-reconnect triggered by {command}')
                    self.connect()
                except Exception as e:
                    logger.error(f'{self._label} {command} -> RECONNECT FAILED: {e}')
                    return None
            if self.driver is None:
                return None

            cmd_upper = command.strip().upper()
            stream = command.encode('utf-8') + b'\n'

            # Save and configure timeouts. Per-readline timeout is small
            # so we loop quickly when no data; total wall-clock budget is
            # `timeout`. inter_byte_timeout lets readline tolerate gaps
            # within a line that exceed our per-call timeout.
            saved_timeout = self.driver.timeout
            saved_inter = getattr(self.driver, 'inter_byte_timeout', None)
            self.driver.timeout = 0.2
            try:
                self.driver.inter_byte_timeout = self._V35_INTER_BYTE_TIMEOUT
            except (ValueError, AttributeError):
                # Some serial backends reject setting inter_byte_timeout
                # — non-fatal, fall back to plain readline timeout.
                pass

            t_start = time.monotonic()
            try:
                # Flush stale bytes from any previous response that may
                # have arrived after exchange_command's drain.
                stale = self.driver.in_waiting
                if stale > 0:
                    discarded = self.driver.read(stale)
                    _serial_log_warn = logger.warning if stale > 16 else logger.debug
                    _serial_log_warn(
                        f'{self._label} v3.5 pre-write stale={stale}B: {discarded!r}')

                self.driver.write(stream)

                while time.monotonic() - t_start < timeout:
                    line = self.driver.readline().decode('utf-8', 'ignore').strip()
                    if not line:
                        # Per-readline timeout fired (no \n). Check total
                        # budget; loop continues if we still have time.
                        continue
                    if line.startswith('RE:') or line.upper() == cmd_upper:
                        # RE: echo, drain and read next line.
                        continue
                    # First non-RE: line is the terminator (spec R2').
                    # Zero-sleep in_waiting check for stragglers (event
                    # lines, late-arriving bytes from a wedged firmware,
                    # etc.). Should be 0 in healthy state.
                    stragglers = self.driver.in_waiting
                    if stragglers > 0:
                        extra = self.driver.read(stragglers)
                        logger.warning(
                            f'{self._label} v3.5 unexpected post-terminator '
                            f'bytes after {cmd_upper}: {extra!r}')
                    return line

                # Total wall-clock timeout.
                logger.warning(
                    f'{self._label} v3.5 exchange({command}) timeout '
                    f'after {timeout}s')
                return None

            except Exception as e:
                logger.error(
                    f'{self._label} v3.5 exchange({command}) exception: {e}')
                self._close_driver()
                return None

            finally:
                self.driver.timeout = saved_timeout
                if saved_inter is not None:
                    try:
                        self.driver.inter_byte_timeout = saved_inter
                    except (ValueError, AttributeError):
                        pass

    def exchange_command(self, command, response_numlines=1, timeout=None,
                         stop_on_empty=False):
        """Override: route v3.5 single-call commands through the
        terminator-based read path. Multi-line commands (LED_READ ALL,
        ADC_READ all, SELFTEST, CALIBRATE, DIAG, CHIP_CHECK, BOOT_LOG,
        STIM_STOP ALL with multiple active) should call
        exchange_multiline directly — that path already does
        terminator-based read.

        v3.0.x path falls through to the base class behavior unchanged.
        """
        if (self._use_v35()
                and response_numlines == 1
                and not stop_on_empty
                and getattr(self, 'firmware_silent', False) is not True):
            v35_timeout = timeout if timeout is not None else 2.0
            return self._exchange_v35(command, timeout=v35_timeout)
        return super().exchange_command(
            command, response_numlines=response_numlines,
            timeout=timeout, stop_on_empty=stop_on_empty)

    def _safety_leds_off(self):
        """Turn off all LEDs immediately after connect (thermal safety).

        Uses synchronous `exchange_command` rather than fire-and-forget
        because v3.5's two-line ack (`RE: LED_OFF` + `OK`) leaves
        residue in the serial buffer that races the next caller's INFO/
        STATUS/etc. Bench-confirmed 2026-04-27 (SN 7162-19): fire-and-
        forget LED_OFF caused get_info() to return 'OK' on the first call
        post-connect. v3.0.x had a 1-line LEDS_OFF response so the race
        was tighter and rarely hit; v3.5's strict framing makes the race
        deterministic.

        Best-effort: if the board doesn't respond within 0.5 s, log and
        continue. The board may be in a state where it can't process
        commands; the safety call is a defense-in-depth, not load-bearing.
        """
        try:
            command = self._build_leds_off_cmd()
            self.exchange_command(command, timeout=0.5)
            logger.info('[LED Class ] Safety LEDS_OFF sent on connect')
        except Exception as e:
            logger.warning(f'[LED Class ] Safety LEDS_OFF failed: {e}')

    def _on_disconnect(self):
        """Clear LED state cache on disconnect (called under self._lock)."""
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
        logger.info('[LED Class ] LED state cache cleared on disconnect')

    def _connect_bench_callables(self):
        """Driver methods benched at connect-time (release gate §2.3).

        `get_info` dispatches to v3.0.x multi-line INFO or v3.5 single-
        line INFO, giving the core cross-firmware latency comparison
        point for the LED board.
        """
        return [('get_info', self.get_info)]

    # ------------------------------------------------------------------
    # Protocol detection
    # ------------------------------------------------------------------
    def _use_v35(self):
        """True iff the connected board speaks v3.5 short-text protocol
        AND advertises the `led` capability. Capability-probe is preferred
        over version-string comparison (FIRMWARE_PROTOCOL.md §1).

        Defensive getattr() matches the firmware_silent pattern in
        SerialBoard — tests construct LEDBoard via __new__ (bypassing
        __init__) and set only the fields they need. protocol_version is
        set in SerialBoard.__init__; if it hasn't run, assume LEGACY.
        """
        if getattr(self, 'protocol_version', None) != ProtocolVersion.V35:
            return False
        return 'led' in getattr(self, 'features', [])

    # ------------------------------------------------------------------
    # Emergency-halt
    # ------------------------------------------------------------------
    def stop(self):
        """Emergency-halt all LED activity — aborts async ops + STIM + off.

        v3.5: real `STOP` command. Firmware aborts SELFTEST/CALIBRATE,
        clears STIM trains FIRST so the Timer ISR can't re-drive the DAC,
        then drives all channels to dark, restores enables for the next
        command sequence. Returns single-line `STOPPED`.

        LEGACY (v3.0.x): no STOP command in firmware. v3.0.x has no STIM
        capability, so `leds_off()` is the safe equivalent — there's no
        ISR to race. Degrades to `self.leds_off()` and annotates the
        result so the caller can see what actually happened.

        Returns a normalized dict so callers don't branch on protocol:
            {'ok': bool, 'stopped': bool,
             'response': raw_response_str | None,
             'note': str | None}

        Returns None on driver error.
        """
        if self._use_v35():
            resp = self.exchange_command('STOP', timeout=5)
            if resp is None:
                return None
            return {
                'ok': 'STOPPED' in (resp or ''),
                'stopped': 'STOPPED' in (resp or ''),
                'response': resp,
                'note': None,
            }

        # LEGACY: no STOP command; fall back to leds_off.
        self.leds_off()
        return {
            'ok': True,
            'stopped': True,
            'response': None,
            'note': 'LEGACY v3.0.x: degraded to leds_off (no STIM to abort)',
        }

    # ------------------------------------------------------------------
    # Color / channel mapping
    # ------------------------------------------------------------------
    _COLOR_TO_CH = {
        'Blue': 0, 'Green': 1, 'Red': 2,
        'BF': 3, 'PC': 4, 'DF': 5,
    }

    _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}

    def color2ch(self, color):
        """Convert color name to numerical channel."""
        return self._COLOR_TO_CH.get(color, 3)

    def ch2color(self, channel):
        """Convert numerical channel to color name."""
        return self._CH_TO_COLOR.get(channel, 'BF')

    def available_channels(self):
        return tuple(self._COLOR_TO_CH.values())

    def available_colors(self):
        return tuple(self._COLOR_TO_CH.keys())

    # ------------------------------------------------------------------
    # Bulk enable / disable
    # ------------------------------------------------------------------
    def leds_enable(self):
        if self._use_v35():
            resp = self.exchange_command('LED_ENABLE ALL')
            if resp is None:
                logger.warning('[LED Class ] leds_enable() v3.5 no response')
            return
        response = self.exchange_command('LEDS_ENT')
        if response is None:
            logger.warning('[LED Class ] leds_enable() got no response')

    def leds_disable(self):
        if self._use_v35():
            resp = self.exchange_command('LED_DISABLE ALL')
            if resp is not None:
                with self._state_lock:
                    for color in self.led_ma:
                        self.led_ma[color] = -1
            else:
                logger.warning('[LED Class ] leds_disable() v3.5 no response')
            return
        response = self.exchange_command('LEDS_ENF')
        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
        else:
            logger.warning('[LED Class ] leds_disable() got no response')

    # ------------------------------------------------------------------
    # Status / wait
    # ------------------------------------------------------------------
    def get_status(self):
        """Query board async-op state + active STIM channels.

        v3.5: real STATUS command — single-line `STATUS idle=<0|1>
        op=<NONE|SELFTEST|CALIBRATE> elapsed_ms=<N>
        stim=<ch:emit/target,...|->`. Returns parsed dict.

        LEGACY: not implemented in v3.0.x LED firmware. Returns None.
        Callers can probe via has_feature('status') before calling.
        """
        if not self._use_v35() or not self.has_feature('status'):
            return None
        resp = self.exchange_command('STATUS')
        if resp is None:
            return None
        return self._parse_v35_status_line(resp)

    @staticmethod
    def _parse_v35_status_line(line):
        """Parse `STATUS idle=N op=X elapsed_ms=N stim=spec` to dict.

        stim spec is `ch:emit/target,ch:emit/target` or `-` for none.
        Each entry yields {'ch':int, 'pulses_emitted':int,
        'pulses_remaining':int, 'count_target':int}.
        """
        if not line or not line.startswith('STATUS '):
            return None
        result = {}
        for token in line.split()[1:]:
            if '=' not in token:
                continue
            k, v = token.split('=', 1)
            result[k] = v
        if 'idle' in result:
            result['idle'] = result['idle'] == '1'
        if 'elapsed_ms' in result:
            try:
                result['elapsed_ms'] = int(result['elapsed_ms'])
            except ValueError:
                pass
        stim_raw = result.get('stim', '-')
        if stim_raw == '-':
            result['stim'] = []
        else:
            stim_list = []
            for entry in stim_raw.split(','):
                if ':' in entry and '/' in entry:
                    ch_part, rest = entry.split(':', 1)
                    em_part, tgt_part = rest.split('/', 1)
                    try:
                        em = int(em_part)
                        tgt = int(tgt_part)
                        stim_list.append({
                            'ch': int(ch_part),
                            'pulses_emitted': em,
                            'pulses_remaining': max(0, tgt - em),
                            'count_target': tgt,
                        })
                    except ValueError:
                        pass
            result['stim'] = stim_list
        return result

    def wait_until_on(self, timeout: float = 5.0):
        """Poll STATUS until any running async op completes or times out.

        LEGACY: returns immediately (STATUS not implemented in v3.0.x).
        v3.5: polls STATUS every 100 ms until idle==True or timeout.
        """
        if not self._use_v35() or not self.has_feature('status'):
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_status()
            if status is None:
                return
            if status.get('idle') is True:
                return
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Cached state accessors
    # ------------------------------------------------------------------
    def get_led_ma(self, color):
        with self._state_lock:
            return self.led_ma.get(color, -1)

    def is_led_on(self, color) -> bool:
        with self._state_lock:
            return self.led_ma.get(color, -1) > 0

    def get_led_state(self, color) -> dict:
        with self._state_lock:
            mA = self.led_ma.get(color, -1)
            enabled = mA > 0
        return {
            'enabled': enabled,
            'illumination': mA,
        }

    def get_led_states(self) -> dict:
        with self._state_lock:
            snapshot = {color: {'enabled': mA > 0, 'illumination': mA}
                        for color, mA in self.led_ma.items()}
        return snapshot

    # ------------------------------------------------------------------
    # Per-channel control. Safety limits are defense-in-depth at driver
    # level — the API layer (lumascope_api.py) also validates, but the
    # driver enforces independently in case of direct calls.
    # ------------------------------------------------------------------
    _MAX_CHANNEL = 5
    _MAX_MA = 1000  # Firmware CH_MAX — absolute hardware limit

    def _validate_led_args(self, channel, mA):
        if not (0 <= int(channel) <= self._MAX_CHANNEL):
            raise ValueError(
                f"LED channel {channel} out of range [0-{self._MAX_CHANNEL}]")
        if not (0 <= int(mA) <= self._MAX_MA):
            raise ValueError(
                f"LED current {mA} mA out of safe range [0-{self._MAX_MA}]")

    def _build_led_on_cmd(self, channel, mA):
        """Validate args; return (color, command_string) for active protocol."""
        self._validate_led_args(channel, mA)
        color = self.ch2color(channel=channel)
        if self._use_v35():
            cmd = f'LED_SET {int(channel)} {int(mA)}'
        else:
            cmd = f'LED{int(channel)}_{int(mA)}'
        return color, cmd

    def _build_led_off_cmd(self, channel):
        color = self.ch2color(channel=channel)
        if self._use_v35():
            cmd = f'LED_OFF {int(channel)}'
        else:
            cmd = f'LED{int(channel)}_OFF'
        return color, cmd

    def _build_leds_off_cmd(self):
        return 'LED_OFF ALL' if self._use_v35() else 'LEDS_OFF'

    def _update_state_cache(self, color: str, mA):
        """Update the cached LED state under lock."""
        with self._state_lock:
            self.led_ma[color] = mA

    def led_on(self, channel, mA, block=False, timeout: float = 5.0):
        """Turn on LED at channel number at mA power.

        If block=True, verify correct callback before returning (with
        timeout). The block= contract is preserved across protocols.
        """
        color, command = self._build_led_on_cmd(channel, mA)
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, mA)
        else:
            logger.warning(
                f'[LED Class ] led_on(ch={channel}, mA={mA}) got no response')

        if block:
            if self._use_v35():
                # v3.5: response is the post-RE: status line (`OK` on
                # success). `OK` in the response confirms commit; no need
                # to substring-match the channel/mA again.
                if 'OK' in (response or ''):
                    return
                deadline = time.monotonic() + timeout
                while response is None or 'OK' not in (response or ''):
                    if time.monotonic() > deadline:
                        logger.warning(
                            f'[LED Class ] led_on(ch={channel}, mA={mA}, '
                            f'block=True) timed out after {timeout}s')
                        break
                    time.sleep(0.01)
                    response = self.exchange_command(command)
                    if response is not None:
                        self._update_state_cache(color, mA)
                return

            # v3.0.x: legacy substring-verify behavior (preserves the
            # original block= contract for sealed field units).
            def _check_each_substr(substrings, result):
                for sub_str in substrings:
                    if sub_str not in result:
                        return False
                return True

            deadline = time.monotonic() + timeout
            while response is None or (
                    command not in response and not _check_each_substr(
                        ['LED', str(int(channel)), str(int(mA))], response)):
                if time.monotonic() > deadline:
                    logger.warning(
                        f'[LED Class ] led_on(ch={channel}, mA={mA}, '
                        f'block=True) timed out after {timeout}s')
                    break
                time.sleep(0.01)
                response = self.exchange_command(command)
                if response is not None:
                    self._update_state_cache(color, mA)

    def led_off(self, channel):
        """Turn off LED at channel number."""
        color, command = self._build_led_off_cmd(channel)
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, -1)
        else:
            logger.warning(
                f'[LED Class ] led_off(ch={channel}) got no response')

    def led_on_fast(self, channel, mA):
        """Fast write-only version of led_on for time-critical toggling.

        Bench (2026-04-20) showed host-side scheduling is unreliable
        below ~20 ms pulse width, which is why STIM moved into firmware
        — this fast path stays for non-stim use cases (short flash
        during capture, warm-up pulses).
        """
        color, command = self._build_led_on_cmd(channel, mA)
        self._update_state_cache(color, mA)
        self._write_command_fast(command)

    def led_off_fast(self, channel):
        """Fast write-only version of led_off for time-critical toggling."""
        color, command = self._build_led_off_cmd(channel)
        self._update_state_cache(color, -1)
        self._write_command_fast(command)

    def leds_off(self):
        """Turn off all LEDs."""
        command = self._build_leds_off_cmd()
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
        else:
            logger.warning('[LED Class ] leds_off() got no response')

    def leds_off_fast(self):
        """Fast write-only version to turn off all LEDs."""
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
        self._write_command_fast(self._build_leds_off_cmd())

    # ------------------------------------------------------------------
    # Engineering mode (v3.0.x only — removed in v3.5 per
    # FIRMWARE_PROTOCOL.md §3 and primary-session decision 2026-04-21:
    # accident prevention moves to UI layer, not modal firmware state).
    # ------------------------------------------------------------------
    def enter_engineering_mode(self, timeout=5.0):
        """Enter engineering mode on LEGACY firmware. v3.5 returns True
        immediately without a wire command; host UI is responsible for
        gating destructive operations behind user confirmation.
        """
        if self._use_v35():
            logger.info(
                '[LED Class ] enter_engineering_mode() no-op on v3.5 '
                '(engineering mode removed)')
            return True

        resp = self.exchange_multiline(
            'FACTORY', timeout=timeout,
            end_markers=['Y/N', 'y/n', 'FACTORY'])
        if resp is None:
            logger.warning('[LED Class ] enter_engineering_mode(): no response')
            return False
        if 'Y/N' not in resp.upper():
            logger.warning(
                f'[LED Class ] enter_engineering_mode(): no Y/N prompt in: {resp!r}')
            return False
        confirm_resp = self.exchange_multiline(
            'Y', timeout=timeout,
            end_markers=['FACTORY', 'Engineering', 'RAW', 'ADC'])
        time.sleep(0.5)
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Entered engineering mode')
        return True

    def exit_engineering_mode(self):
        """Exit engineering mode back to safe mode (Q command). No-op on v3.5."""
        if self._use_v35():
            return None
        resp = self.exchange_command('Q', timeout=3)
        time.sleep(0.3)
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Exited engineering mode')
        return resp

    # ------------------------------------------------------------------
    # SELFTEST
    # ------------------------------------------------------------------
    def selftest(self, timeout=180):
        """Run LED SELFTEST and return parsed result lines.

        v3.5: multi-line streaming, ends `SELFTEST_DONE tested=N
        skipped=M aborted=<0|1> slow=<0|1>`.
        v3.0.x: multi-line text scraped for per-channel ramp results,
        ends `Complete`/`COMPLETE`/`DONE`/`ERROR`.

        Returns list of result line strings; empty list on failure.
        """
        end_markers = (['SELFTEST_DONE'] if self._use_v35()
                       else ['Complete', 'COMPLETE', 'DONE', 'ERROR'])
        resp = self.exchange_multiline(
            'SELFTEST', timeout=timeout, end_markers=end_markers)
        if resp is None:
            logger.warning('[LED Class ] selftest(): no response')
            return []
        lines = [line.strip() for line in resp.split('\n') if line.strip()]
        logger.info(f'[LED Class ] selftest(): {len(lines)} lines')
        return lines

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------
    def get_info(self):
        """Send INFO and return parsed dict.

        v3.5: parse single-line `INFO ver=X date=Y sub=LED proto=3.5
        serial=Z cal=N reset=R heap=N adc=ok:0xN ref=V features=a,b,c`.
        v3.0.x: parse multi-line text response for version/date.

        Returns dict with at minimum 'raw' (full response). v3.5 path
        adds 'version', 'date', 'cal_status', 'features' (list),
        'sub'/'proto'/'serial'/'reset'/'heap'/'adc'/'ref' fields. Empty
        dict on failure.
        """
        if self._use_v35():
            resp = self.exchange_command('INFO', timeout=2)
            return self._parse_v35_info_dict(resp) if resp else {}

        resp = self.exchange_command('INFO', response_numlines=6, timeout=2)
        if resp is None:
            return {}
        if isinstance(resp, list):
            raw = '\n'.join(resp)
        else:
            raw = resp
        result = {'raw': raw}
        ver_match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', raw)
        if ver_match:
            result['version'] = ver_match.group(1)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', raw)
        if date_match:
            result['date'] = date_match.group(1)
        if 'Cal:' in raw or 'Calibrated' in raw:
            result['cal_status'] = 'calibrated' if 'Calibrated' in raw else 'default'
        return result

    @staticmethod
    def _parse_v35_info_dict(line):
        """v3.5 single-line INFO → dict.

        Tokenizer-style parse matching the firmware-side emitter in
        `Firmware-FW4.0/LED Controller/main.py:_handle_info`. Order is
        canonical and stable but the parser is not order-sensitive.
        """
        if not line:
            return {}
        if not line.startswith('INFO '):
            return {'raw': line}
        result = {'raw': line}
        for token in line.split()[1:]:
            if '=' in token:
                k, v = token.split('=', 1)
                result[k] = v
        # Normalize commonly-used keys to legacy parser names so
        # downstream consumers see a uniform schema across protocols.
        if 'ver' in result:
            result['version'] = result['ver']
        if 'cal' in result:
            result['cal_status'] = (
                'calibrated' if result['cal'] == '1' else 'default')
        if 'features' in result and isinstance(result['features'], str):
            result['features'] = [f for f in result['features'].split(',') if f]
        return result

    # ------------------------------------------------------------------
    # LED current readback
    # ------------------------------------------------------------------
    def read_led_current(self, channel):
        """Read measured LED current (mA) from ADC feedback.

        v3.5: `LED_READ <ch>` → single-line `LED_READ ch=N mA=X
        v_sens=Y v_ledk=Z`. No engineering-mode gating.
        v3.0.x: requires v2.0+ firmware in engineering mode; multi-line
        text response with `I_SENS` line carrying the mA value.

        Returns measured current in mA (float), or None on
        error/unsupported.
        """
        if self._use_v35() and self.has_feature('led'):
            resp = self.exchange_command(f'LED_READ {int(channel)}')
            if not resp:
                return None
            for token in resp.split():
                if token.startswith('mA='):
                    try:
                        return float(token.split('=', 1)[1])
                    except ValueError:
                        return None
            return None

        if not self.is_v2:
            return None
        command = f'LEDREAD{int(channel)}'
        try:
            # Firmware sends: echo (handled by exchange_command),
            # I_SENS line, LED_K line.
            lines = self.exchange_command(command, response_numlines=3)
            if lines is None:
                return None
            for line in lines:
                if 'I_SENS' in line and 'mA' in line:
                    m = re.search(r'([\d.]+)\s*mA', line)
                    if m:
                        return float(m.group(1))
        except Exception as e:
            logger.error(
                f'[LED Class ] read_led_current({channel}) failed: {e}')
        return None

    # ------------------------------------------------------------------
    # Firmware-side STIM pulse trains (v3.5+ only).
    #
    # No v3.0.x equivalent. v3.0.x firmware's "STIM" stopgap was the
    # 2026-04-20 bench fix on the v3.0.8-firmware-stim branch and is not
    # exposed through this driver (host-side stim runs through
    # led_on_fast / led_off_fast edges on v3.0.x). The v3.5 firmware-
    # owned pulse timing retires host-side scheduling for anything
    # under ~20 ms pulse widths.
    #
    # Callers must check supports_firmware_stim() before calling. On
    # LEGACY boards these log a warning and return None.
    # ------------------------------------------------------------------
    def firmware_stim(self, channel, mA, pulse_ms, period_ms, count):
        """Start a firmware-owned single-channel pulse train.

        v3.5 wire: `STIM <ch> <mA> <pulse_ms> <period_ms> <count>` →
        single-line `STIM_RUN ch=N pulse_us=X period_us=Y count=Z`
        (after `RE: STIM` echo).

        Returns dict with parsed `kind`/`ch`/`pulse_us`/`period_us`/
        `count` fields on success, None on LEGACY/failure. Poll
        get_status() for count progress (events deferred to FW4.1 per
        FIRMWARE_PROTOCOL.md §3.6).
        """
        if not self.supports_firmware_stim():
            logger.warning(
                '[LED Class ] firmware_stim() called but stim feature '
                'not advertised')
            return None
        cmd = (f'STIM {int(channel)} {float(mA)} {float(pulse_ms)} '
               f'{float(period_ms)} {int(count)}')
        resp = self.exchange_command(cmd, timeout=2)
        if not resp:
            return None
        return self._parse_kv_line(resp)

    def firmware_stim_stop(self, channel='ALL'):
        """Stop one channel or all STIM trains.

        v3.5 wire: `STIM_STOP <ch|ALL>`. Single channel → single-line
        `STIM_STOPPED ch=N pulses=M`. ALL with multiple active channels
        → multi-line, ends `END_STIM_STOP`.
        """
        if not self.supports_firmware_stim():
            logger.warning(
                '[LED Class ] firmware_stim_stop() called but stim '
                'not advertised')
            return None
        ch_arg = 'ALL' if channel == 'ALL' else str(int(channel))
        if ch_arg == 'ALL':
            return self.exchange_multiline(
                f'STIM_STOP {ch_arg}', timeout=5,
                end_markers=['END_STIM_STOP', 'STIM_STOPPED'])
        return self.exchange_command(f'STIM_STOP {ch_arg}', timeout=5)

    def supports_firmware_stim(self):
        """True iff the connected firmware advertises the stim capability
        (v3.5+). Mirrors the v3.0.8-firmware-stim probe used by LVP's
        StimulationController to decide host-side vs firmware-side stim
        on a per-protocol basis."""
        return self._use_v35() and self.has_feature('stim')

    # ------------------------------------------------------------------
    # Generic v3.5 key=val response parser
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_kv_line(line):
        """`KIND field1=val1 field2=val2 ...` → dict.

        Leading word becomes `kind`. Used for STIM_RUN / STIM_STOPPED /
        any single-line key=val response shape.
        """
        if not line:
            return {}
        tokens = line.split()
        if not tokens:
            return {}
        result = {'kind': tokens[0]}
        for token in tokens[1:]:
            if '=' in token:
                k, v = token.split('=', 1)
                result[k] = v
        return result
