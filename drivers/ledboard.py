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
        # to the board (measured 62°C). New v3.0.4+ firmware initializes LEDs
        # off on boot, but this guard protects against old firmware and
        # interrupted previous sessions.
        self._safety_leds_off()

    def _safety_leds_off(self):
        """Turn off all LEDs immediately after connect (thermal safety).

        Uses fire-and-forget write to minimize delay. If the board doesn't
        respond, this is a best-effort attempt — the board may be in a
        state where it can't process commands.

        Both LEGACY and V4 paths covered. On FW4.0 this goes through
        exchange_json with a short timeout so the call doesn't block
        connect() if the board is slow to respond.
        """
        try:
            if self.protocol_version == ProtocolVersion.V4:
                self.exchange_json({'cmd': 'LED_OFF', 'ch': 'ALL'}, timeout=0.5)
            else:
                self._write_command_fast('LEDS_OFF')
            logger.info('[LED Class ] Safety LEDS_OFF sent on connect')
        except Exception as e:
            logger.warning(f'[LED Class ] Safety LEDS_OFF failed: {e}')

    def _on_disconnect(self):
        """Clear LED state cache on disconnect (called under self._lock)."""
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
        logger.info('[LED Class ] LED state cache cleared on disconnect')

    _COLOR_TO_CH = {
        'Blue': 0, 'Green': 1, 'Red': 2,
        'BF': 3, 'PC': 4, 'DF': 5,
    }

    _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}

    def color2ch(self, color):
        """ Convert color name to numerical channel """
        return self._COLOR_TO_CH.get(color, 3)

    def ch2color(self, channel):
        """ Convert numerical channel to color name """
        return self._CH_TO_COLOR.get(channel, 'BF')

    def available_channels(self):
        return tuple(self._COLOR_TO_CH.values())

    def available_colors(self):
        return tuple(self._COLOR_TO_CH.keys())

    # interperet commands
    # ------------------------------------------
    # board status: 'STATUS' case insensitive
    # LED enable:   'LED' channel '_ENT' where channel is numbers 0 through 5, or S (plural/all)
    # LED disable:  'LED' channel '_ENF' where channel is numbers 0 through 5, or S (plural/all)
    # LED on:       'LED' channel '_MA' where channel is numbers 0 through 5, or S (plural/all)
    #                and MA is numerical representation of mA
    # LED off:      'LED' channel '_OFF' where channel is numbers 0 through 5, or S (plural/all)

    # ------------------------------------------------------------------
    # Dual-protocol dispatch.
    # LEGACY (pre-FW4.0 firmware): existing text commands (LED_, LEDS_OFF,
    #   LED0_100, ...). Permanent support lane per primary-session posture
    #   (2026-04-21) — sealed field units may stay on this forever.
    # V4 (FW4.0 firmware): JSON-object commands over exchange_json, gated
    #   by has_feature('led'). Enables optional id correlation + future
    #   push-event integration.
    # Both paths are first-class; don't remove one in favor of the other.
    # ------------------------------------------------------------------
    def _use_v4(self):
        """True iff the connected board speaks FW4.0 AND advertises led.
        Capability-probe (features[]) is preferred over version-string
        comparison per primary-session guidance.

        Defensive getattr() matches the firmware_silent pattern in
        SerialBoard — tests construct LEDBoard via __new__ (bypassing
        __init__) and set only the fields they need. protocol_version is
        set in SerialBoard.__init__; if it hasn't run, assume LEGACY."""
        if getattr(self, 'protocol_version', None) != ProtocolVersion.V4:
            return False
        return 'led' in getattr(self, 'features', [])

    def leds_enable(self):
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LED_ENABLE', 'ch': 'ALL'})
            if resp is None or resp.get('ok') is not True:
                logger.warning('[LED Class ] leds_enable() V4 path no/bad response')
            return
        command = 'LEDS_ENT'
        response = self.exchange_command(command)
        if response is None:
            logger.warning('[LED Class ] leds_enable() got no response')

    def leds_disable(self):
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LED_DISABLE', 'ch': 'ALL'})
            if resp is not None and resp.get('ok') is True:
                with self._state_lock:
                    for color in self.led_ma:
                        self.led_ma[color] = -1
            else:
                logger.warning('[LED Class ] leds_disable() V4 path no/bad response')
            return
        command = 'LEDS_ENF'
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
        else:
            logger.warning('[LED Class ] leds_disable() got no response')

    def get_status(self):
        """Query board async-op state + active STIM channels.

        FW4.0 (V4): real STATUS command — returns {idle, op?, elapsed_ms?,
        stim?[], ...}. Host uses this to watch HOME/CALIBRATE/SELFTEST
        progress and STIM pulse counts.

        LEGACY: not implemented in v3.0.x LED firmware. Returns None.
        This was documented as "do not use" in the v3.0.x driver; on FW4.0
        the command exists and works, so callers can now probe via
        has_feature('status') before calling.
        """
        if self._use_v4() and self.has_feature('status'):
            return self.exchange_json({'cmd': 'STATUS'})
        return None

    def wait_until_on(self, timeout: float = 5.0):
        """Poll STATUS until any running async op completes or times out.

        LEGACY: returns immediately (STATUS not implemented in v3.0.x).
        V4: polls STATUS every 100 ms until idle==True or timeout.
        """
        if not self._use_v4() or not self.has_feature('status'):
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self.exchange_json({'cmd': 'STATUS'})
            if resp is None:
                return
            if resp.get('idle') is True:
                return
            time.sleep(0.1)

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

    # Safety limits — defense-in-depth validation at driver level.
    # The API layer (lumascope_api.py) also validates, but the driver
    # must enforce independently in case of direct calls.
    _MAX_CHANNEL = 5
    _MAX_MA = 1000  # Firmware CH_MAX — absolute hardware limit

    def _validate_and_build_led_cmd(self, channel, mA):
        """Validate channel/mA and return (color, command) string.

        Shared by led_on() and led_on_fast() to eliminate duplicate validation.
        """
        if not (0 <= int(channel) <= self._MAX_CHANNEL):
            raise ValueError(f"LED channel {channel} out of range [0-{self._MAX_CHANNEL}]")
        if not (0 <= int(mA) <= self._MAX_MA):
            raise ValueError(f"LED current {mA} mA out of safe range [0-{self._MAX_MA}]")
        color = self.ch2color(channel=channel)
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        return color, command

    def _update_state_cache(self, color: str, mA):
        """Update the cached LED state under lock."""
        with self._state_lock:
            self.led_ma[color] = mA

    def led_on(self, channel, mA, block=False, timeout: float = 5.0):
        """
        Turn on LED at channel number at mA power
        If block=True, verify correct callback before returning (with timeout)
        """
        color, command = self._validate_and_build_led_cmd(channel, mA)

        if self._use_v4():
            # FW4.0 LED_SET is self-healing — every call asserts enable pin
            # and writes DAC (firmware _set_led_state). block= is a no-op
            # on V4 because the response carries the commanded mA, so the
            # caller doesn't need to re-poll to verify.
            resp = self.exchange_json({'cmd': 'LED_SET', 'ch': int(channel), 'mA': float(mA)})
            if resp is not None and resp.get('ok') is True:
                self._update_state_cache(color, mA)
            else:
                logger.warning(f'[LED Class ] led_on(ch={channel}, mA={mA}) V4 no/bad response: {resp}')
            return

        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, mA)
        else:
            logger.warning(f'[LED Class ] led_on(ch={channel}, mA={mA}) got no response')

        def check_each_substr(substrings, result):
            for sub_str in substrings:
                if sub_str not in result:
                    return False
            return True

        if block:
            deadline = time.monotonic() + timeout
            while response is None or (command not in response and not check_each_substr(['LED', str(int(channel)), str(int(mA))], response)):
                if time.monotonic() > deadline:
                    logger.warning(f'[LED Class ] led_on(ch={channel}, mA={mA}, block=True) timed out after {timeout}s')
                    break
                time.sleep(0.01)  # Prevent busy-wait CPU burn
                response = self.exchange_command(command)
                if response is not None:
                    self._update_state_cache(color, mA)

    def led_off(self, channel):
        """ Turn off LED at channel number """
        color = self.ch2color(channel=channel)

        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LED_OFF', 'ch': int(channel)})
            if resp is not None and resp.get('ok') is True:
                self._update_state_cache(color, -1)
            else:
                logger.warning(f'[LED Class ] led_off(ch={channel}) V4 no/bad response: {resp}')
            return

        command = 'LED' + str(int(channel)) + '_OFF'
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, -1)
        else:
            logger.warning(f'[LED Class ] led_off(ch={channel}) got no response')

    def led_on_fast(self, channel, mA):
        """Fast write-only version of led_on for time-critical toggling.
        V4 uses exchange_json with a short timeout; LEGACY uses the write-
        only fast path for lowest possible host-side latency. On FW4.0
        the bench measurement (2026-04-20) showed host-side scheduling is
        unreliable below ~20 ms pulse width, which is why STIM moved into
        firmware — this fast path stays for non-stim use cases (short
        flash during capture, warm-up pulses, etc.)."""
        color, command = self._validate_and_build_led_cmd(channel, mA)
        self._update_state_cache(color, mA)

        if self._use_v4():
            # Timeout kept small — caller is optimizing for throughput, not
            # retry semantics. Response discarded but events still flushed.
            self.exchange_json({'cmd': 'LED_SET', 'ch': int(channel), 'mA': float(mA)},
                               timeout=0.5)
            return

        self._write_command_fast(command)

    def led_off_fast(self, channel):
        """Fast write-only version of led_off for time-critical toggling."""
        color = self.ch2color(channel=channel)
        self._update_state_cache(color, -1)
        command = 'LED' + str(int(channel)) + '_OFF'

        if self._use_v4():
            self.exchange_json({'cmd': 'LED_OFF', 'ch': int(channel)}, timeout=0.5)
            return

        self._write_command_fast(command)

    def leds_off(self):
        """ Turn off all LEDs """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LED_OFF', 'ch': 'ALL'})
            if resp is not None and resp.get('ok') is True:
                with self._state_lock:
                    for color in self.led_ma:
                        self.led_ma[color] = -1
            else:
                logger.warning(f'[LED Class ] leds_off() V4 no/bad response: {resp}')
            return

        command = 'LEDS_OFF'
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

        if self._use_v4():
            self.exchange_json({'cmd': 'LED_OFF', 'ch': 'ALL'}, timeout=0.5)
            return

        command = 'LEDS_OFF'
        self._write_command_fast(command)

    # ------------------------------------------------------------------
    # Engineering mode and diagnostics
    # ------------------------------------------------------------------
    def enter_engineering_mode(self, timeout=5.0):
        """Enter engineering mode on LEGACY firmware. FW4.0 killed the
        concept of modal engineering mode (primary-session decision
        2026-04-21 — accident prevention moves to UI layer). On V4 this
        returns True immediately without a wire command; host UI is
        responsible for gating destructive operations (CALIBRATE, DAC_RAW
        etc.) behind a user confirmation before issuing the command.
        """
        if self._use_v4():
            logger.info('[LED Class ] enter_engineering_mode() no-op on V4 (engineering mode removed)')
            return True

        resp = self.exchange_multiline(
            'FACTORY', timeout=timeout,
            end_markers=['Y/N', 'y/n', 'FACTORY'])
        if resp is None:
            logger.warning('[LED Class ] enter_engineering_mode(): no response')
            return False
        if 'Y/N' not in resp.upper():
            logger.warning(f'[LED Class ] enter_engineering_mode(): no Y/N prompt in: {resp!r}')
            return False
        # Confirm with Y
        confirm_resp = self.exchange_multiline(
            'Y', timeout=timeout,
            end_markers=['FACTORY', 'Engineering', 'RAW', 'ADC'])
        # Drain any remaining help text
        time.sleep(0.5)
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Entered engineering mode')
        return True

    def exit_engineering_mode(self):
        """Exit engineering mode on LEGACY. No-op on V4."""
        if self._use_v4():
            return None
        resp = self.exchange_command('Q', timeout=3)
        time.sleep(0.3)
        # Drain any remaining output
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Exited engineering mode')
        return resp

    def selftest(self, timeout=180):
        """Run LED SELFTEST and return parsed results.

        LEGACY path: multi-line text response scraped for per-channel
        ramp results.

        V4 path: exchange_json returns a structured dict:
            {"ok": True, "channels": [{"ch":N,"present":bool,"levels":[...]},...],
             "channels_tested": N, "channels_skipped": M, ...}
        We return the structured response directly on V4 — callers that
        care about the old line-list format can format it themselves.
        Runtime 10-30s on fast, 60-120s on slow; honor the caller timeout.
        """
        if self._use_v4():
            return self.exchange_json({'cmd': 'SELFTEST'}, timeout=timeout)

        resp = self.exchange_multiline(
            'SELFTEST', timeout=timeout,
            end_markers=['Complete', 'COMPLETE', 'DONE', 'ERROR'])
        if resp is None:
            logger.warning('[LED Class ] selftest(): no response')
            return []
        lines = [line.strip() for line in resp.split('\n') if line.strip()]
        logger.info(f'[LED Class ] selftest(): {len(lines)} lines')
        return lines

    def get_info(self):
        """Return a dict describing firmware identity + capabilities.

        LEGACY: parses the multi-line text response for version/date.
        V4: returns the structured INFO dict directly — includes
            features[] for capability probing, serial/chip_id, heap_free,
            reset_cause, boot_log[].
        """
        if self._use_v4():
            return self.exchange_json({'cmd': 'INFO'}) or {}

        resp = self.exchange_command('INFO', response_numlines=6, timeout=2)
        if resp is None:
            return {}
        if isinstance(resp, list):
            raw = '\n'.join(resp)
        else:
            raw = resp
        result = {'raw': raw}
        # Parse version
        import re as _re
        ver_match = _re.search(r'v(\d+\.\d+(?:\.\d+)?)', raw)
        if ver_match:
            result['version'] = ver_match.group(1)
        date_match = _re.search(r'(\d{4}-\d{2}-\d{2})', raw)
        if date_match:
            result['date'] = date_match.group(1)
        if 'Cal:' in raw or 'Calibrated' in raw:
            result['cal_status'] = 'calibrated' if 'Calibrated' in raw else 'default'
        return result

    def read_led_current(self, channel):
        """Read measured LED current (mA) via I_SENS ADC.

        LEGACY: requires v2.0+ firmware in engineering mode. Parses the
        LEDREAD<ch> multi-line text response.
        V4: exchange_json LED_READ returns {"mA": float, "v_sens": ...,
            "v_ledk": ...}; no engineering-mode gating on FW4.0.
        Returns measured current in mA, or None on error/unsupported.
        """
        if self._use_v4():
            resp = self.exchange_json({'cmd': 'LED_READ', 'ch': int(channel)})
            if resp is not None and resp.get('ok') is True:
                return resp.get('mA')
            return None

        if not self.is_v2:
            return None
        command = f'LEDREAD{int(channel)}'
        try:
            # Firmware sends: echo (handled by exchange_command), I_SENS line, LED_K line
            lines = self.exchange_command(command, response_numlines=3)
            if lines is None:
                return None
            # Parse I_SENS line: "LED0 I_SENS  (AIN14): 1.2800V  ->   200.1 mA"
            for line in lines:
                if 'I_SENS' in line and 'mA' in line:
                    m = re.search(r'([\d.]+)\s*mA', line)
                    if m:
                        return float(m.group(1))
        except Exception as e:
            logger.error(f'[LED Class ] read_led_current({channel}) failed: {e}')
        return None

    # ------------------------------------------------------------------
    # FW4.0-only helpers for firmware-side STIM pulse trains.
    #
    # These methods are NEW in FW4.0 — no LEGACY equivalent. v3.0.x
    # firmware's "STIM" stopgap was the 2026-04-20 bench fix on the
    # v3.0.8-firmware-stim branch and is not exposed through this driver
    # (host-side stim controller runs through led_on_fast / led_off_fast
    # edges on v3.0.x). The FW4.0 firmware-owned pulse timing retires
    # host-side scheduling for anything under ~20 ms pulse widths.
    #
    # Callers must check has_feature('stim') before calling. On LEGACY
    # boards these log a warning and return None.
    # ------------------------------------------------------------------
    def firmware_stim(self, channel, mA, pulse_ms, period_ms, count):
        """Start a firmware-owned single-channel pulse train. Returns the
        RUNNING response ({"ok":True,"ch":N,"status":"RUNNING",...}) on
        success; None on LEGACY or failure. Poll STATUS for count progress
        or subscribe to on_event for the stim_done completion event."""
        if not self._use_v4() or not self.has_feature('stim'):
            logger.warning('[LED Class ] firmware_stim() called but stim feature not advertised')
            return None
        return self.exchange_json({
            'cmd': 'STIM',
            'ch': int(channel),
            'mA': float(mA),
            'pulse_ms': float(pulse_ms),
            'period_ms': float(period_ms),
            'count': int(count),
        })

    def firmware_stim_stop(self, channel='ALL'):
        """Stop one channel or all STIM trains. Self-heals off structurally."""
        if not self._use_v4() or not self.has_feature('stim'):
            logger.warning('[LED Class ] firmware_stim_stop() called but stim not advertised')
            return None
        ch_arg = 'ALL' if channel == 'ALL' else int(channel)
        return self.exchange_json({'cmd': 'STIM_STOP', 'ch': ch_arg})

    def supports_firmware_stim(self):
        """True iff the connected firmware advertises the stim capability
        (FW4.0+). Mirrors the v3.0.8-firmware-stim probe method used by
        LVP's StimulationController to decide host-side vs firmware-side
        stim on a per-protocol basis."""
        return self._use_v4() and self.has_feature('stim')
