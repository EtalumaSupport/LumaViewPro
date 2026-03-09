# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Protocol imaging time estimator.

Estimates total protocol execution time by modeling all timing components:
stage movement, LED switching, camera exposure, autofocus, and overhead.
"""

import math
from dataclasses import dataclass, field
from datetime import timedelta

from lvp_logger import logger
from modules.objectives_loader import ObjectiveLoader


# ---------------------------------------------------------------------------
# Stage timing constants (from motor controller / simulator)
# ---------------------------------------------------------------------------
XY_SPEED_UM_PER_S = 50_000.0       # ~50 mm/s
Z_SPEED_UM_PER_S = 5_000.0         # ~5 mm/s
Z_BACKLASH_UM = 25.0               # overshoot distance for downward Z moves
SERIAL_CMD_OVERHEAD_S = 0.003       # ~3ms per serial round-trip

# ---------------------------------------------------------------------------
# LED / camera constants
# ---------------------------------------------------------------------------
LED_SETTLE_S = 0.005                # 5ms LED settle after led_on
LED_SERIAL_S = 0.003                # ~3ms serial round-trip for LED command
FRAME_DRAIN_COUNT = 2               # stale frames drained after parameter change
MIN_FRAME_TIME_S = 0.05             # minimum frame time (50ms)
AUTOGAIN_MAX_DURATION_S = 1.0       # default autogain max wait
VIDEO_MAX_FPS = 40                  # capture loop cap

# ---------------------------------------------------------------------------
# Overhead
# ---------------------------------------------------------------------------
STEP_OVERHEAD_S = 0.020             # misc serial commands, polling, loop sleeps
AF_STEP_OVERHEAD_S = 0.015          # per-AF-step overhead (polling, sleep)
AF_GREASE_REDISTRIBUTION_INTERVAL = 100  # every 100 AFs, full Z cycle


@dataclass
class StepTimeEstimate:
    """Timing breakdown for a single protocol step."""
    step_index: int
    step_name: str
    move_time_s: float = 0.0
    led_time_s: float = 0.0
    autofocus_time_s: float = 0.0
    autogain_time_s: float = 0.0
    capture_time_s: float = 0.0
    overhead_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.move_time_s + self.led_time_s + self.autofocus_time_s
                + self.autogain_time_s + self.capture_time_s + self.overhead_s)


@dataclass
class ScanTimeEstimate:
    """Timing breakdown for one complete scan (all steps)."""
    step_estimates: list[StepTimeEstimate] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(s.total_s for s in self.step_estimates)

    @property
    def num_steps(self) -> int:
        return len(self.step_estimates)

    @property
    def num_autofocus_steps(self) -> int:
        return sum(1 for s in self.step_estimates if s.autofocus_time_s > 0)

    @property
    def num_video_steps(self) -> int:
        return sum(1 for s in self.step_estimates
                   if s.capture_time_s > 0 and s.step_name)  # approximation


@dataclass
class ProtocolTimeEstimate:
    """Full protocol timing estimate."""
    scan_estimate: ScanTimeEstimate
    num_scans: int
    period_s: float
    duration_s: float
    estimated_total_s: float
    scan_fits_in_period: bool
    scan_overrun_s: float

    @property
    def estimated_completion(self) -> timedelta:
        return timedelta(seconds=self.estimated_total_s)

    def summary(self) -> str:
        scan_td = timedelta(seconds=self.scan_estimate.total_s)
        total_td = self.estimated_completion
        period_td = timedelta(seconds=self.period_s)
        lines = [
            f"Steps per scan: {self.scan_estimate.num_steps}",
            f"Scan time: {scan_td}",
            f"Period: {period_td}",
            f"Number of scans: {self.num_scans}",
            f"Estimated total: {total_td}",
        ]
        if not self.scan_fits_in_period:
            lines.append(
                f"WARNING: Scan exceeds period by "
                f"{timedelta(seconds=self.scan_overrun_s)}")
        af_count = self.scan_estimate.num_autofocus_steps
        if af_count > 0:
            lines.append(f"Autofocus steps: {af_count}")
        return "\n".join(lines)


class ProtocolTimeEstimator:
    """Estimates total imaging time for a protocol."""

    def __init__(self, objectives_loader: ObjectiveLoader = None):
        if objectives_loader is None:
            try:
                objectives_loader = ObjectiveLoader()
            except Exception:
                objectives_loader = None
        self._objectives = objectives_loader

    def estimate(self, protocol) -> ProtocolTimeEstimate:
        """Estimate total protocol execution time."""
        scan_estimate = self._estimate_scan(protocol)

        period_s = protocol.period().total_seconds()
        duration_s = protocol.duration().total_seconds()
        num_scans = max(1, int(duration_s / period_s)) if period_s > 0 else 1

        scan_time = scan_estimate.total_s
        scan_fits = scan_time <= period_s
        overrun = max(0.0, scan_time - period_s)
        effective_period = max(scan_time, period_s)
        estimated_total = num_scans * effective_period

        return ProtocolTimeEstimate(
            scan_estimate=scan_estimate,
            num_scans=num_scans,
            period_s=period_s,
            duration_s=duration_s,
            estimated_total_s=estimated_total,
            scan_fits_in_period=scan_fits,
            scan_overrun_s=overrun,
        )

    def _estimate_scan(self, protocol) -> ScanTimeEstimate:
        """Estimate one complete scan through all protocol steps."""
        steps = protocol.steps()
        if steps is None or len(steps) == 0:
            return ScanTimeEstimate()

        estimates = []
        prev_step = None

        for idx, step in steps.iterrows():
            obj_info = self._get_objective_info(step.get('Objective', ''))
            est = self._estimate_step(idx, step, prev_step, obj_info)
            estimates.append(est)
            prev_step = step

        return ScanTimeEstimate(step_estimates=estimates)

    def _estimate_step(
        self,
        idx: int,
        step,
        prev_step,
        objective_info: dict | None,
    ) -> StepTimeEstimate:
        """Estimate time for a single protocol step."""
        name = str(step.get('Name', f'Step {idx + 1}'))
        exposure_s = float(step.get('Exposure', 50)) / 1000.0

        # Movement
        move_time = self._estimate_movement(step, prev_step)

        # LED switching + settle
        led_time = LED_SETTLE_S + LED_SERIAL_S

        # Autofocus
        af_time = 0.0
        if step.get('Auto_Focus', False) and objective_info:
            af_time = self._estimate_autofocus(objective_info, exposure_s)

        # Auto-gain wait
        ag_time = 0.0
        if step.get('Auto_Gain', False):
            ag_time = AUTOGAIN_MAX_DURATION_S

        # Capture
        acquire = step.get('Acquire', 'image')
        if acquire == 'video':
            vc = step.get('Video Config', {})
            duration = float(vc.get('duration', 5.0)) if isinstance(vc, dict) else 5.0
            capture_time = duration + max(exposure_s, MIN_FRAME_TIME_S)
        else:
            frame_time = max(exposure_s, MIN_FRAME_TIME_S)
            frame_drain = FRAME_DRAIN_COUNT * frame_time
            sum_count = max(1, int(step.get('Sum', 1)))
            capture_time = frame_drain + exposure_s * sum_count

        return StepTimeEstimate(
            step_index=idx,
            step_name=name,
            move_time_s=move_time,
            led_time_s=led_time,
            autofocus_time_s=af_time,
            autogain_time_s=ag_time,
            capture_time_s=capture_time,
            overhead_s=STEP_OVERHEAD_S,
        )

    @staticmethod
    def _estimate_movement(step, prev_step) -> float:
        """Estimate XYZ movement time between steps."""
        if prev_step is None:
            return 0.0

        dx = abs(float(step.get('X', 0)) - float(prev_step.get('X', 0)))
        dy = abs(float(step.get('Y', 0)) - float(prev_step.get('Y', 0)))
        dz = abs(float(step.get('Z', 0)) - float(prev_step.get('Z', 0)))

        xy_time = max(dx, dy) / XY_SPEED_UM_PER_S if max(dx, dy) > 0 else 0.0
        z_time = dz / Z_SPEED_UM_PER_S if dz > 0 else 0.0

        # Z overshoot penalty for downward moves
        z_overshoot = 0.0
        cur_z = float(prev_step.get('Z', 0))
        tgt_z = float(step.get('Z', 0))
        if cur_z > tgt_z and tgt_z > (Z_BACKLASH_UM + 50):
            z_overshoot = (2 * Z_BACKLASH_UM) / Z_SPEED_UM_PER_S

        # XY and Z move concurrently; Z overshoot is sequential
        return max(xy_time, z_time) + z_overshoot

    def _estimate_autofocus(
        self,
        objective_info: dict,
        exposure_s: float,
    ) -> float:
        """Estimate autofocus duration in seconds."""
        af_range = float(objective_info.get('AF_range', 75))
        af_max = float(objective_info.get('AF_max', 36))
        af_min = float(objective_info.get('AF_min', 8))

        if af_max <= 0 or af_min <= 0:
            return 0.0

        # Count total steps across all passes
        total_steps = 0
        resolution = af_max
        sweep_range = 2 * af_range  # first pass covers full range

        while resolution > af_min:
            n_steps = int(sweep_range / resolution) + 1
            total_steps += n_steps
            sweep_range = 2 * resolution
            resolution = resolution / 3.0

        # Final pass at af_min
        n_steps = int(sweep_range / af_min) + 1
        total_steps += n_steps

        # Time per AF step: Z move + frame grab + overhead
        frame_time = max(exposure_s, 0.075)
        time_per_step = (af_max / Z_SPEED_UM_PER_S) + frame_time + AF_STEP_OVERHEAD_S

        # Initial move to z_min and final move to best position
        initial_move_time = af_range / Z_SPEED_UM_PER_S

        return total_steps * time_per_step + 2 * initial_move_time

    def _get_objective_info(self, objective_id: str) -> dict | None:
        """Look up objective parameters."""
        if not self._objectives or not objective_id:
            return None
        try:
            return self._objectives.get_objective_info(objective_id=objective_id)
        except Exception:
            return None
