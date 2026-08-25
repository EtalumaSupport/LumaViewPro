# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IDS Peak hardware tests -- opt-in via --run-ids-hardware.

These tests require:
  1. Real ids_peak SDK installed (not the conftest MagicMock)
  2. A connected IDS camera

Skipped by default. Run with:
    pytest tests/test_ids_hardware.py --run-ids-hardware

The `ids_hardware` marker is gated by conftest.pytest_collection_modifyitems --
test bodies do NOT need their own skip dance.
"""

import time
import unittest

import numpy as np
import pytest

# When --run-ids-hardware is set, conftest skips installing the ids_peak
# mock so the real SDK can load here. When the flag is NOT set, this
# import succeeds against the conftest mock and the marker below skips
# the tests at collection time.
from drivers.idscamera import IDSCamera


@pytest.mark.ids_hardware
class TestIDS(unittest.TestCase):
    def setUp(self):
        self.camera = IDSCamera()
        self.camera.open_and_start()

    def tearDown(self):
        self.camera.disconnect()
        time.sleep(1)

    def test_connect_disconnect(self):
        self.assertTrue(self.camera.disconnect())
        self.assertTrue(self.camera.connect())

    def test_grab(self):
        self.assertTrue(self.camera.is_grabbing())
        self.camera.stop_grabbing()
        self.assertFalse(self.camera.is_grabbing())

    def test_frame_size(self):
        # On-grid request: delivered exactly (no surplus to crop).
        self.camera.set_frame_size(1920, 1528)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1920, 'height': 1528})
        # Off-grid width: oversize-then-crop now delivers the EXACT request
        # (acquire 1920x1532, crop to 1919x1529); previously silent-floored to
        # 1872x1528. get_frame_size reports the delivered (cropped) size.
        self.camera.set_frame_size(1919, 1529)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1919, 'height': 1529})
        # Off-increment on both axes: also delivered exactly (was 1440x904).
        self.camera.set_frame_size(1480, 906)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1480, 'height': 906})

    def test_pixel_format(self):
        formats = self.camera.get_supported_pixel_formats()
        self.camera.set_pixel_format(formats[0])
        self.assertTrue(self.camera.get_pixel_format() in formats)

    def test_pixel_format_logical_mono8(self):
        # Logical 'Mono8' must resolve to whichever Mono8-class entry the
        # camera actually exposes (Mono8, Mono8g, etc). Skipped on cameras
        # without any Mono8-class entry (e.g. Sony IMX676 in U3-34L0XCP-M
        # exposes only Mono10g40IDS / Mono12g24IDS) -- on those, the resolver
        # correctly returns None and set_pixel_format reports unsupported,
        # tested by the pure-logic suite TestIDSPixelFormatResolver.
        supported = self.camera.get_supported_pixel_formats()
        if not any(s.startswith('Mono8') for s in supported):
            self.skipTest(f'camera has no Mono8-class entry (supported={list(supported)})')
        self.assertTrue(self.camera.set_pixel_format('Mono8'))
        active = self.camera.get_pixel_format()
        self.assertTrue(
            active.startswith('Mono8'), f'Logical Mono8 resolved to non-Mono8 entry: {active}'
        )

    def test_exposure_t(self):
        self.camera.exposure_t(15)
        self.assertAlmostEqual(self.camera.get_exposure_t(), 15.0, delta=0.01)

    def test_binning_size(self):
        self.assertTrue(self.camera.set_binning_size(1))
        self.assertEqual(self.camera.get_binning_size(), 1)
        self.assertTrue(self.camera.set_binning_size(2))
        self.assertEqual(self.camera.get_binning_size(), 2)
        self.assertFalse(self.camera.set_binning_size(3))
        self.assertEqual(self.camera.get_binning_size(), 2)
        self.assertFalse(self.camera.set_binning_size(0))
        self.assertEqual(self.camera.get_binning_size(), 2)

    def test_binned_aoi_space_max_halves(self):
        """Resolve whether the AOI maxima are reported in binned or native px.

        The oversize-then-crop framing math is planned in displayed
        (post-binning) pixel space, which is correct only if Width/Height
        maxima are in binned pixels -- i.e. they roughly halve from 1x to 2x
        binning. If the max does NOT shrink, the AOI is in native pixels and
        the framing plan must divide once at the end instead. This records the
        fact so the math can be trusted; run with -s to see the printed values.
        """
        self.camera.set_binning_size(1)
        max_1x = self.camera.get_max_frame_size()
        self.camera.set_binning_size(2)
        max_2x = self.camera.get_max_frame_size()
        self.camera.set_binning_size(1)  # restore
        print(f'[binned-aoi-space] max 1x={max_1x} 2x={max_2x}')
        self.assertLess(
            max_2x['width'],
            max_1x['width'],
            f'AOI max did not shrink with binning ({max_1x} -> {max_2x}): AOI is in '
            'native pixels, so the oversize-crop plan must divide by binning at the end',
        )
        self.assertAlmostEqual(max_2x['width'], max_1x['width'] / 2, delta=max_1x['width'] * 0.02)

    def test_grab_frame(self):
        time.sleep(1)  # Allow time for the camera to start grabbing
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        # The grabbed array's row count must match the delivered frame height
        # (assert against the configured size, not a hard-coded resolution that
        # goes stale the moment the default frame size changes).
        self.assertEqual(len(self.camera.array), self.camera.get_frame_size()['height'])
        self.assertIsNotNone(timestamp)

    def _feature_node(self, group, name):
        """Read one optional-feature node's probe record (presence / access /
        value / current) back off the live camera via the production diagnostic
        snapshot. ``group`` is 'remote' or 'stream'."""
        snap = self.camera.read_diagnostic_snapshot(duration_s=0.2)
        return snap['feature_nodes'].get(group, {}).get(name, {})

    def test_set_test_pattern_write(self):
        """The TestPattern setter writes on hardware (node presence + ReadWrite
        was already bench-proven; this proves the write lands). TestPattern is a
        remote-nodemap node the SDK locks during acquisition (TLParamsLocked), so
        the write is attempted live first, then -- if the stream lock rejected it
        -- re-applied with acquisition stopped, mirroring the transport-param
        setters. The while-grabbing result is recorded for the operator; the
        stopped-stream write is the deterministic confirmation.
        """
        tp = self._feature_node('remote', 'TestPattern')
        if not tp.get('present'):
            self.skipTest(f'TestPattern absent on this body (probe={tp})')
        original = tp.get('current')

        self.camera.set_test_pattern(enabled=True, pattern='ColorBar')
        applied_while_grabbing = (
            self._feature_node('remote', 'TestPattern').get('current') == 'ColorBar'
        )

        if applied_while_grabbing:
            self._exercise_test_pattern_entries()
        else:
            # Locked during acquisition -- release the stream lock and retry.
            self.camera.stop_grabbing()
            try:
                self.assertTrue(
                    self.camera.set_test_pattern(enabled=True, pattern='ColorBar'),
                    'TestPattern write rejected even with the stream stopped',
                )
                self.assertEqual(
                    self._feature_node('remote', 'TestPattern').get('current'), 'ColorBar'
                )
                self._exercise_test_pattern_entries()
            finally:
                self.camera.start_grabbing()

        # Leave the camera as we found it.
        if original and original != 'Off':
            self.camera.set_test_pattern(enabled=True, pattern=original)

        print(
            f'[test-pattern-write] applied_while_grabbing={applied_while_grabbing} '
            f'needs_stopped_stream={not applied_while_grabbing}'
        )

    def _exercise_test_pattern_entries(self):
        """A bogus entry is rejected (False, not a raise); disabling restores Off."""
        self.assertFalse(self.camera.set_test_pattern(enabled=True, pattern='NotAPattern'))
        self.assertTrue(self.camera.set_test_pattern(enabled=False))
        self.assertEqual(self._feature_node('remote', 'TestPattern').get('current'), 'Off')

    def test_set_max_transfer_size_write(self):
        self._assert_stream_param_write(
            'U3vStreamChannelBulkTransferSize', self.camera.set_max_transfer_size
        )

    def test_set_num_max_queued_urbs_write(self):
        self._assert_stream_param_write(
            'U3vStreamChannelTransferRequestCount', self.camera.set_num_max_queued_urbs
        )

    def _assert_stream_param_write(self, node_name, setter):
        """Confirm a DataStream channel-parameter setter writes, and record
        whether a stopped stream is required. These nodes are transport-layer
        params bracketed by TLParamsLocked (set 1 in start_grabbing, 0 in
        stop_grabbing), so a write is expected to be rejected mid-stream and to
        succeed once acquisition is stopped. The deterministic assertion is the
        stopped-stream write; the while-grabbing result is recorded for the
        operator (run with -s) since firmware may or may not allow it live.
        """
        node = self._feature_node('stream', node_name)
        if not node.get('present'):
            self.skipTest(f'{node_name} absent on this body (probe={node})')
        original = node.get('value')

        applied_while_grabbing = setter(int(original))

        # Stop acquisition to release the TLParamsLocked transport lock, then the
        # write must take on a ReadWrite node.
        self.camera.stop_grabbing()
        try:
            self.assertTrue(
                setter(int(original)), f'{node_name}: write rejected even with the stream stopped'
            )
            read_back = self._feature_node('stream', node_name).get('value')
            self.assertEqual(read_back, original)
        finally:
            self.camera.start_grabbing()

        print(
            f'[stream-param-write] {node_name} value={original} '
            f'applied_while_grabbing={applied_while_grabbing} '
            f'applied_when_stopped=True (needs_stopped_stream='
            f'{not applied_while_grabbing})'
        )

    def test_gain(self):
        self.camera.gain(10)
        self.assertAlmostEqual(self.camera.get_gain(), 10.0, delta=0.1)

    def test_gain_node_is_linear_multiplier_not_native_db(self):
        """Anti-fragility guard for the dB<->factor gain conversion.

        LVP's gain model is dB (shared with the Pylon driver); the IDS driver
        assumes the camera's Gain node is a LINEAR multiplier and converts
        dB = 20*log10(factor) (idscamera _query_dynamic_capabilities / gain /
        get_gain_db). SFNC permits a body to express Gain natively in dB -- such a
        body would be double-converted. This records the node's unit + range
        (run with -s to read them) and fails loudly if the body reports gain in
        dB, which is the exact signal to branch the conversion on the unit.
        """
        nm = self.camera.remote_nodemap
        selector = self.camera._resolve_gain_selector()
        if selector:
            nm.FindNode('GainSelector').SetCurrentEntry(selector)
        gain_node = nm.FindNode('Gain')

        minimum = gain_node.Minimum()
        maximum = gain_node.Maximum()
        value = gain_node.Value()
        try:
            unit = gain_node.Unit()
        except Exception as e:
            unit = f'<unavailable: {e}>'
        try:
            reported_db = self.camera.get_gain()
        except Exception as e:
            reported_db = f'<error: {e}>'
        print(
            f'[gain-node] selector={selector} unit={unit!r} value={value} '
            f'min={minimum} max={maximum} get_gain_db()={reported_db} dB'
        )

        # A unit string saying dB is the authoritative tell that the node is
        # native dB and the 20*log10(factor) conversion double-converts.
        if isinstance(unit, str) and 'db' in unit.strip().lower():
            self.fail(
                f'Gain node reports unit {unit!r} (native dB): the driver applies '
                '20*log10(factor) and would double-convert on this body -- branch '
                'gain()/get_gain_db()/_query_dynamic_capabilities on the unit.'
            )

        # Structural backstop when no unit string is exposed: a linear analog
        # gain multiplier floors at ~1.0x (1x = 0 dB); a native-dB node floors
        # near 0.0 (0 dB). A minimum well below 1.0 means the node is NOT a
        # linear multiplier and the log10 conversion is wrong for this body.
        self.assertGreaterEqual(
            minimum,
            0.9,
            f'Gain.Minimum()={minimum} is not ~1.0x: the node is not a linear '
            'multiplier (likely native dB), so dB=20*log10(factor) double-converts.',
        )

    def test_native_depth_frame_is_right_aligned_uint16(self):
        """Keystone bench check: a grabbed frame arrives at the sensor's native
        depth in a uint16 container, right-aligned.

        Right-aligned means full scale is (1 << significant_bits) - 1 and every
        pixel fits in the low significant_bits of the uint16. If the SDK
        left-aligns instead, pixel values ride in the high bits and overflow
        that range -- the assertion below catches it, settling the #1
        undocumented unpack unknown. drivers/ids_unpack.py is the cross-check
        for the expected layout.

        Needs light on the sensor: a near-dark frame can't distinguish the two
        alignments (small values stay small either way), so the test skips
        rather than pass vacuously when the frame is too dark.

        Selects the Mono12 native wire format explicitly: the driver defaults the
        IMX676 body to Mono10g40IDS (8-bit delivery) for live speed, so without
        this the grab returns uint8 and the native-depth (uint16) right-alignment
        this test exists to check is never exercised. The actual 12-bit packed
        entry is taken from the live supported list (not the logical 'Mono12',
        which falls back to Mono10 on a body with no 12-bit format); skip a body
        that advertises none.
        """
        supported = self.camera.get_supported_pixel_formats()
        native12 = next((f for f in supported if f.startswith('Mono12')), None)
        if native12 is None:
            self.skipTest(f'body advertises no Mono12 native format (supported={list(supported)})')
        self.assertTrue(self.camera.set_pixel_format(native12), f'could not set {native12}')

        time.sleep(1)  # let the grab loop store a native-depth frame post-reconfigure
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertIsNotNone(timestamp)

        frame = self.camera.array
        self.assertEqual(frame.dtype, np.uint16, 'native-depth frame must be a uint16 container')

        sig = self.camera.last_significant_bits
        self.assertEqual(sig, 12, f'Mono12 native wire must deliver significant_bits=12; got {sig}')

        peak = int(frame.max())
        if peak < 256:
            self.skipTest(
                f'frame too dark (max={peak}) to judge alignment -- put light on the sensor'
            )
        full_scale = (1 << sig) - 1
        self.assertLessEqual(
            peak,
            full_scale,
            f'frame.max()={peak} exceeds {full_scale}: the SDK is NOT right-aligned at '
            f'significant_bits={sig} (values ride in the high bits) -- the consuming code '
            f'must shift. This is the alignment unknown the rebuild flagged.',
        )
        self.assertGreaterEqual(int(frame.min()), 0)

    def test_unpack_benchmark(self):
        """Head-to-head: the SDK ConvertTo unpack vs the numpy ids_unpack path,
        on real packed frames. Two questions in one run:

          1. Correctness -- ConvertTo is the oracle. A zero-mismatch result
             proves the derived packed-bit layout (and right-alignment) is
             correct, since IDS documents the layout only as a figure.
          2. Speed -- the per-frame timings show whether the numpy unpack is
             actually faster than ConvertTo on this host (the ~18 fps display
             cap is the ConvertTo throughput).

        Run with output visible:
            pytest tests/test_ids_hardware.py -k unpack_benchmark --run-ids-hardware -s

        Needs light on the sensor so a real (non-zero) image exercises the
        full-scale bits; an all-dark frame would match trivially on both paths.
        """
        time.sleep(1)  # let acquisition settle
        res = self.camera.benchmark_unpack(n_frames=200)

        print('\n[IDS unpack benchmark]')
        for k in (
            'wire_format',
            'width',
            'height',
            'available_formats',
            'packed_dtype',
            'n_compared',
            'mismatches',
            'first_mismatch',
            'convert',
            'numpy',
            'icv',
        ):
            print(f'  {k}: {res.get(k)}')

        self.assertGreater(res['n_compared'], 0, f'no frames were compared: {res.get("error")}')
        self.assertEqual(
            res['mismatches'],
            0,
            f'numpy unpack disagrees with ConvertTo on {res["mismatches"]}/'
            f'{res["n_compared"]} frames -- the derived layout is wrong: '
            f'{res.get("first_mismatch")}',
        )

    def test_afl_packed_process_acceptance(self):
        """Bench probe: does ids_peak_afl.Manager.Process() accept a PACKED
        image, or must it be fed an unpacked one?

        The IMX676 body has no camera-side auto-exposure/gain; a host auto
        routine must run through ids_peak_afl. Process() takes an IPL image,
        but IDS does not document whether the packed Mono10g40IDS/Mono12g24IDS
        wire formats are accepted. This answers it empirically so the future
        auto routine knows whether it can feed the raw buffer or must unpack
        each frame first. The probe also dumps the real ids_peak_afl API
        surface (module/Manager/Controller) since the Python binding spelling
        is not pinned. Informational -- reads the answer, does not pass/fail on
        it.

        Run with output visible:
            pytest tests/test_ids_hardware.py -k afl_packed --run-ids-hardware -s --driver-log
        """
        time.sleep(1)  # let acquisition settle
        res = self.camera.probe_afl_packed_acceptance()

        print('\n[AFL packed-Process probe]')
        for k in sorted(res):
            print(f'  {k}: {res.get(k)}')

        if not res.get('afl_importable'):
            self.skipTest(f'ids_peak_afl not importable: {res.get("afl_import_errors")}')

        # The headline datapoint: whether the PACKED image was accepted. Not an
        # assertion -- either answer is valid and printed above. Fail only if the
        # probe could not set up the manager at all (nothing was learned).
        packed = res.get('packed')
        print(
            f'  ==> PACKED accepted: {packed.get("accepted") if packed else None} '
            f'(exception={packed.get("exception") if packed else None}); '
            f'UNPACKED accepted: {(res.get("unpacked") or {}).get("accepted")}'
        )
        self.assertTrue(
            res.get('packed') is not None
            or any(k in res for k in ('error', 'lib_init_error', 'manager_setup_error')),
            f'AFL probe returned neither a packed result nor a recorded reason: {res}',
        )

    def test_camera_capability_nodemap_probe(self):
        """Read-only bench probe: dump the presence/access/entries of the nodes
        behind our open vendor questions, so they can be answered from the
        camera instead of by email.

        Covers low-light/HDR/conversion-gain and NoiseReduction (is any
        sensor-level low-light mode exposed?), the hardware-auto nodes (confirm
        ExposureAuto/GainAuto/BalanceWhiteAuto are absent), the DataStream
        statistics + StreamBufferHandlingMode entries, and the temperature
        selector set. Pure enumeration -- writes nothing to the camera.

        Run with output visible:
            pytest tests/test_ids_hardware.py -k capability_nodemap --run-ids-hardware -s
        """

        def _probe_node(nm, name):
            try:
                if not nm.HasNode(name):
                    return f'{name}: ABSENT'
            except Exception as e:
                return f'{name}: HasNode error {type(e).__name__}: {e}'
            try:
                node = nm.FindNode(name)
            except Exception as e:
                return f'{name}: PRESENT, FindNode error {type(e).__name__}: {e}'
            parts = [f'{name}: PRESENT']
            for meth in ('IsReadable', 'IsWriteable'):
                try:
                    parts.append(f'{meth[2:].lower()}={getattr(node, meth)()}')
                except Exception:
                    pass
            try:
                parts.append(f'entries={[e.SymbolicValue() for e in node.AvailableEntries()]}')
            except Exception:
                pass
            return ' '.join(parts)

        from lvp_logger import logger as _cap_log

        def _emit(line):
            # Print for the -s terminal AND log through the driver logger so the
            # result is captured by --driver-log (a bare print() is dropped from
            # the collected log bundle).
            print(line)
            _cap_log.info(f'[CAM Class ] capability-probe: {line.strip()}')

        remote = self.camera.remote_nodemap
        remote_nodes = [
            # hardware auto (expect ABSENT on this body)
            'ExposureAuto',
            'GainAuto',
            'BalanceWhiteAuto',
            # low-light / HDR / conversion-gain / noise (any sensor-level mode?)
            'SensorOperationMode',
            'SensorShutterMode',
            'GainConversionMode',
            'ConversionGain',
            'HDRMode',
            'HDREnable',
            'NoiseReduction',
            # black level (long-exposure drift)
            'BlackLevel',
            'BlackLevelAuto',
            'BlackLevelSelector',
            # temperature selector set
            'DeviceTemperature',
            'DeviceTemperatureSelector',
            # throughput component (bench-confirmed RO; recheck on 3.93)
            'DeviceLinkThroughputLimit',
            'DeviceLinkThroughputLimitComponent',
        ]
        _emit('[capability probe -- remote (SFNC) nodemap]')
        for name in remote_nodes:
            _emit(f'  {_probe_node(remote, name)}')

        try:
            ds_nm = self.camera.data_stream.NodeMaps()[0]
        except Exception as e:
            _emit(f'  <data-stream nodemap unavailable: {type(e).__name__}: {e}>')
            ds_nm = None
        if ds_nm is not None:
            ds_nodes = [
                'StreamBufferHandlingMode',
                'StreamAnnouncedBufferCount',
                'StreamDeliveredFrameCount',
                'StreamDroppedFrameCount',
                'StreamLostFrameCount',
                'StreamIncompleteFrameCount',
                'BufferStatusMonitoringEnabled',
            ]
            _emit('[capability probe -- data-stream nodemap]')
            for name in ds_nodes:
                _emit(f'  {_probe_node(ds_nm, name)}')

        # Sanity: the remote nodemap read path works (PixelFormat is always there).
        self.assertTrue(remote.HasNode('PixelFormat'))

    def test_8bit_direct_unpack_matches_rescale(self):
        """Bench gate for the direct 10->8 delivery (8-bit mode): the SDK's
        packed->Mono8 ConvertTo must match the prior native-then-host-rescale
        within 1 LSB on real frames.

        8-bit image mode now unpacks the Mono10 wire straight to 8-bit in one
        pass (no uint16 intermediate, no display-thread downconvert). That trades
        the host's exact linear rescale for the SDK's bit-shift; this asserts the
        swap is <=1 LSB so saved/displayed 8-bit pixels are unchanged in practice.

        Run on a Mono10 wire (8-bit mode) with light on the sensor:
            pytest tests/test_ids_hardware.py -k 8bit_direct --run-ids-hardware -s
        """
        time.sleep(1)  # let acquisition settle
        res = self.camera.crosscheck_8bit_unpack(n_frames=100)

        print('\n[IDS 8-bit direct-unpack cross-check]')
        for k in (
            'wire_format',
            'skipped',
            'n_compared',
            'max_abs_diff',
            'pixels_over_1lsb',
            'error',
        ):
            print(f'  {k}: {res.get(k)}')

        if res.get('skipped'):
            self.skipTest(res['skipped'])  # not a Mono10/8-bit-mode wire

        self.assertGreater(res['n_compared'], 0, f'no frames were compared: {res.get("error")}')
        # max_abs_diff <= 1 already implies pixels_over_1lsb == 0, so one assertion
        # covers the whole-frame bound.
        self.assertLessEqual(
            res['max_abs_diff'],
            1,
            f'direct ConvertTo(Mono8) drifts {res["max_abs_diff"]} LSB from the rescale '
            f'oracle ({res["pixels_over_1lsb"]} pixels over 1 LSB) -- the SDK downconvert is '
            f'not a simple bit-shift; keep the rescale path',
        )


class TestIDSPixelFormatResolver(unittest.TestCase):
    """Pure-logic tests for IDSCamera._resolve_logical_format_name.

    Runs without hardware -- the resolver is a @staticmethod that takes the
    supported-list as a parameter, so we don't need a connected camera or
    SDK access. Regression test for the 2026-05-04 bench bug where
    set_pixel_format('Mono8') silently fell through to formats[0] on cameras
    whose PixelFormat node uses sensor-specific names (Mono10g40IDS etc).
    """

    def test_exact_match(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(IDSCamera._resolve_logical_format_name('Mono8', ('Mono8',)), 'Mono8')

    def test_mono8_prefix_match(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name(
                'Mono8', ('Mono10g40IDS', 'Mono12g24IDS', 'Mono8g')
            ),
            'Mono8g',
        )

    def test_mono8_no_family_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(
            IDSCamera._resolve_logical_format_name('Mono8', ('Mono10g40IDS', 'Mono12g24IDS'))
        )

    def test_mono12_prefix_match(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name('Mono12', ('Mono8', 'Mono12g24IDS')),
            'Mono12g24IDS',
        )

    def test_mono12_falls_back_to_mono10(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name('Mono12', ('Mono8', 'Mono10g40IDS')),
            'Mono10g40IDS',
        )

    def test_camera_native_name_passes_through(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._resolve_logical_format_name('Mono10g40IDS', ('Mono8', 'Mono10g40IDS')),
            'Mono10g40IDS',
        )

    def test_unknown_logical_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(IDSCamera._resolve_logical_format_name('Mono99', ('Mono8',)))

    def test_empty_supported_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(IDSCamera._resolve_logical_format_name('Mono8', ()))


class TestIDSDefaultPixelFormatSelection(unittest.TestCase):
    """Pure-logic tests for IDSCamera._select_default_pixel_format.

    The connect-time default is derived from the camera's live PixelFormat
    entries, not a model-string-matched static profile, so any IDS body (any
    sensor, any advertised format set) selects a valid lowest-bandwidth format
    without a per-model table entry.
    """

    def test_prefers_mono8_when_present(self):
        from drivers.idscamera import IDSCamera

        # A body that exposes Mono8 (e.g. Basler-style) starts there -- lowest
        # bandwidth for the live preview.
        self.assertEqual(
            IDSCamera._select_default_pixel_format(('Mono12g24IDS', 'Mono10g40IDS', 'Mono8')),
            'Mono8',
        )

    def test_imx676_packed_only_picks_mono10(self):
        from drivers.idscamera import IDSCamera

        # No Mono8 (the IMX676 bodies): 10-bit packed beats 12-bit packed.
        self.assertEqual(
            IDSCamera._select_default_pixel_format(('Mono12g24IDS', 'Mono10g40IDS')),
            'Mono10g40IDS',
        )

    def test_mono12_only_picks_mono12(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._select_default_pixel_format(('Mono12g24IDS',)),
            'Mono12g24IDS',
        )

    def test_colour_only_returns_none(self):
        from drivers.idscamera import IDSCamera

        # This driver's pipeline is mono-only; a colour body has no valid
        # default here (returning a Bayer/RGB entry would feed a mosaic to the
        # mono unpack/blit path).
        self.assertIsNone(IDSCamera._select_default_pixel_format(('BayerRG8', 'RGB8')))

    def test_prefers_packed_over_unpacked_same_depth(self):
        from drivers.idscamera import IDSCamera

        # Same 10-bit depth: the packed IDS entry (fewer bytes on the wire) beats
        # the unpacked one, so connect does not double USB bandwidth by default.
        self.assertEqual(
            IDSCamera._select_default_pixel_format(('Mono10', 'Mono10g40IDS')),
            'Mono10g40IDS',
        )

    def test_empty_supported_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(IDSCamera._select_default_pixel_format(()))


class TestIDSGainSelectorSelection(unittest.TestCase):
    """Pure-logic tests for IDSCamera._select_gain_selector_name.

    The analog gain selector entry is resolved from the body's live GainSelector
    enum, not hardcoded to 'AnalogAll', so a body that names it differently still
    gets gain control (the old hardcoded 'AnalogAll' silently failed every gain
    write on such a body).
    """

    def test_preferred_present(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._select_gain_selector_name(('All', 'AnalogAll', 'DigitalAll'), 'AnalogAll'),
            'AnalogAll',
        )

    def test_preferred_absent_picks_analog_variant(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._select_gain_selector_name(('All', 'AnalogGain'), 'AnalogAll'),
            'AnalogGain',
        )

    def test_no_analog_falls_back_to_all(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._select_gain_selector_name(('All', 'DigitalAll'), 'AnalogAll'),
            'All',
        )

    def test_unknown_only_falls_back_to_first(self):
        from drivers.idscamera import IDSCamera

        self.assertEqual(
            IDSCamera._select_gain_selector_name(('Tap1', 'Tap2'), 'AnalogAll'),
            'Tap1',
        )

    def test_empty_returns_none(self):
        from drivers.idscamera import IDSCamera

        self.assertIsNone(IDSCamera._select_gain_selector_name((), 'AnalogAll'))


class TestIDSDefaultProfile(unittest.TestCase):
    """An unrecognized IDS body gets an IDS-shaped fallback profile, not the
    cross-vendor Mono8 'Unknown' default -- so it is still driven as IDS and its
    capability fields are filled from the live nodemap at connect."""

    def test_is_ids_driver_with_empty_capability_fields(self):
        from drivers.camera_profiles import ids_default_profile

        p = ids_default_profile('U3-99XYZ-M')
        self.assertEqual(p.driver, 'ids')
        self.assertEqual(p.model_name, 'U3-99XYZ-M')
        # pixel formats + pixel size are left for the live nodemap to fill;
        # never fabricate a pixel size (would corrupt the micron scale).
        self.assertEqual(p.pixel_formats, [])
        self.assertEqual(p.pixel_size_um, 0.0)
        self.assertEqual(p.alignment, {'width': 2, 'height': 2})


if __name__ == '__main__':
    unittest.main()
