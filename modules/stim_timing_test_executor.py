"""
Stimulation Timing Test Executor
Systematically tests different timing optimization configurations
"""

import pathlib
import datetime
import json
import shutil
import time
import threading
from typing import Dict, Any, List
from lvp_logger import logger
from modules.stim_timing_test_config import StimTestCase, generate_test_cases, get_quick_test_cases, test_case_to_dict


class StimTimingTestExecutor:
    """Manages systematic testing of stimulation timing optimizations"""
    
    def __init__(self, protocol, protocol_settings, test_cases: List[StimTestCase]):
        """
        Args:
            protocol: The Protocol object to run for each test
            protocol_settings: The ProtocolSettings instance that can run protocols
            test_cases: List of test configurations to run
        """
        self.protocol = protocol
        self.protocol_settings = protocol_settings
        self.test_cases = test_cases
        self.test_run_dir = None
        self.current_test_case = None
        self.test_results = []
        self._test_complete_event = threading.Event()
        
        # Progress callbacks
        self.on_test_start = None  # Called when test starts: (test_idx, total_tests, test_name)
        self.on_test_progress = None  # Called during test: (test_idx, total_tests, test_name, status, est_remaining_seconds)
        self.on_test_complete = None  # Called when test completes: (test_idx, total_tests, test_name, duration)
        
        # Timing tracking
        self._test_durations = []
        self._step_timings = {}  # Track average time for each step type
        self._total_steps_completed = 0
        self._protocol_steps_completed = 0  # Track well/step completions during protocol
        self._protocol_steps_total = 0  # Total wells/steps in protocol
        self._protocol_step_times = []  # Individual step/well times
    
    def start_test_suite(self) -> pathlib.Path:
        """
        Start running the full test suite.
        Returns the path to the results directory.
        """
        from settings_init import settings
        
        # Create test run directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_capture_dir = pathlib.Path(settings['live_folder'])
        self.test_run_dir = base_capture_dir / f"stim_timing_tests_{timestamp}"
        self.test_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test configuration
        config_file = self.test_run_dir / "test_configuration.json"
        with open(config_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'num_tests': len(self.test_cases),
                'test_cases': [test_case_to_dict(tc) for tc in self.test_cases]
            }, f, indent=2)
        
        logger.info(f"[STIM TEST] Starting test suite with {len(self.test_cases)} test cases")
        logger.info(f"[STIM TEST] Results will be saved to: {self.test_run_dir}")
        
        self.test_results = []
        
        # Import sequenced_capture_executor here to avoid circular imports
        from modules import sequenced_capture_executor
        
        # Run each test case
        for idx, test_case in enumerate(self.test_cases):
            try:
                logger.info(f"[STIM TEST] Running test {idx + 1}/{len(self.test_cases)}: {test_case.name}")
                
                # Notify test start
                if self.on_test_start:
                    self.on_test_start(idx, len(self.test_cases), test_case.name)
                
                # Apply test configuration BEFORE running protocol
                step_start = time.time()
                self._apply_test_configuration(test_case, sequenced_capture_executor)
                self._record_step_time('config', time.time() - step_start)
                
                # Notify progress with time estimate
                est_remaining = self._estimate_remaining_time(idx, len(self.test_cases))
                if self.on_test_progress:
                    self.on_test_progress(idx, len(self.test_cases), test_case.name, "Configuration applied", est_remaining)
                
                # Run protocol (timing tracked inside _run_single_test)
                result = self._run_single_test(test_case, idx, sequenced_capture_executor)
                self.test_results.append(result)
                
                # Track duration for time estimation
                self._test_durations.append(result['duration_seconds'])
                
                # Calculate average duration
                avg_duration = sum(self._test_durations) / len(self._test_durations)
                
                # Notify test complete
                if self.on_test_complete:
                    self.on_test_complete(idx, len(self.test_cases), test_case.name, result['duration_seconds'])
                
                # Small delay between tests + cleanup
                logger.info(f"[STIM TEST] Test {idx + 1} complete. Cleaning up...")
                
                # Notify progress
                est_remaining = self._estimate_remaining_time(idx, len(self.test_cases))
                if self.on_test_progress:
                    self.on_test_progress(idx, len(self.test_cases), test_case.name, "Cleaning up...", est_remaining)
                
                # Delete videos from this test during the delay
                test_dir = pathlib.Path(result['test_dir'])
                actual_run_dir = pathlib.Path(result['actual_run_dir'])
                
                # Clean up videos from actual run directory
                step_start = time.time()
                videos_deleted = self._cleanup_video_files(actual_run_dir)
                self._record_step_time('cleanup', time.time() - step_start)
                result['videos_deleted_count'] = videos_deleted
                result['avg_test_duration'] = avg_duration
                
                # Update result file with video deletion count
                result_file = test_dir / "test_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.exception(f"[STIM TEST] Error in test {idx + 1} ({test_case.name})")
                # Create a failure result and continue to next test
                result = {
                    'test_case': test_case_to_dict(test_case),
                    'test_dir': str(self.test_run_dir / f"test_{idx:02d}_{test_case.name}"),
                    'actual_run_dir': str(self.test_run_dir / f"test_{idx:02d}_{test_case.name}"),
                    'start_time': datetime.datetime.now().isoformat(),
                    'end_time': datetime.datetime.now().isoformat(),
                    'duration_seconds': 0,
                    'profiling_data_exists': False,
                    'videos_deleted_count': 0,
                    'success': False,
                    'error': str(e)
                }
                self.test_results.append(result)
                # Continue to next test instead of failing entire suite
        
        # Generate comparison report
        comparison_file = self._generate_comparison_report()
        
        logger.info(f"[STIM TEST] Test suite complete! Results in: {self.test_run_dir}")
        
        return comparison_file
    
    def _run_single_test(self, test_case: StimTestCase, test_idx: int, sequenced_capture_executor) -> Dict[str, Any]:
        """Run a single test case by executing the protocol"""
        from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
        
        # Create directory for this test
        test_dir = self.test_run_dir / f"test_{test_idx:02d}_{test_case.name}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test case configuration
        config_file = test_dir / "test_case_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_case_to_dict(test_case), f, indent=2)
        
        start_time = datetime.datetime.now()
        logger.info(f"[STIM TEST] Starting protocol execution for test: {test_case.name}")
        
        # Notify progress
        est_remaining = self._estimate_remaining_time(test_idx, len(self.test_cases))
        if self.on_test_progress:
            self.on_test_progress(test_idx, len(self.test_cases), test_case.name, "Running protocol...", est_remaining)
        
        # Set up completion callback
        self._test_complete_event.clear()
        protocol_start = time.time()
        
        # Get total number of steps for progress tracking
        self._protocol_steps_total = self.protocol_settings._protocol.num_steps()
        self._protocol_steps_completed = 0
        self._protocol_step_times = []
        
        def on_run_complete(**kwargs):
            logger.info(f"[STIM TEST] Protocol complete for test: {test_case.name}")
            self._test_complete_event.set()
        
        def on_step_complete(step_idx, step_name, duration_seconds):
            """Called after each well/step is captured"""
            self._protocol_steps_completed += 1
            self._protocol_step_times.append(duration_seconds)
            
            # Update progress with current estimate
            if self.on_test_progress:
                est_remaining = self._estimate_remaining_time(test_idx, len(self.test_cases))
                status = f"Protocol: {self._protocol_steps_completed}/{self._protocol_steps_total} wells"
                self.on_test_progress(test_idx, len(self.test_cases), test_case.name, status, est_remaining)
        
        # Run protocol using the protocol_settings instance
        # Pass test_dir as parent_dir so sequenced_capture_executor creates run dir inside it
        try:
            # Get required parameters from protocol_settings
            from settings_init import settings
            
            # Call run_sequenced_capture with test_dir as parent_dir to route output correctly
            self.protocol_settings.run_sequenced_capture(
                run_mode=SequencedCaptureRunMode.SINGLE_SCAN,  # Run one scan for testing
                run_trigger_source='stim_timing_test',
                max_scans=1,
                callbacks={'run_complete': on_run_complete, 'step_complete': on_step_complete},
                parent_dir=test_dir,  # Output goes to test-specific directory
            )
            
            # Wait for protocol to complete (with timeout)
            timeout = 300  # 5 minutes max per test
            if not self._test_complete_event.wait(timeout):
                logger.error(f"[STIM TEST] Test {test_case.name} timed out after {timeout}s")
                success = False
            else:
                success = True
                
        except Exception as e:
            logger.exception(f"[STIM TEST] Error running test {test_case.name}")
            success = False
            
        self._record_step_time('protocol', time.time() - protocol_start)
        
        # Notify progress
        if self.on_test_progress:
            self.on_test_progress(test_idx, len(self.test_cases), test_case.name, "Collecting results...")
        
        # 4. Collect profiling data
        collect_start = time.time()
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Find the actual run directory created by sequenced_capture_executor
        # With parent_dir=test_dir, it creates timestamped dir directly inside test_dir
        actual_run_dir = test_dir
        
        # Look for timestamped directories inside test_dir
        run_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
        if run_dirs:
            actual_run_dir = run_dirs[-1]  # Use most recent timestamped directory
            logger.info(f"[STIM TEST] Found run directory: {actual_run_dir}")
        
        # Collect profiling data from the actual run directory
        profiling_dir = actual_run_dir / "stimulation_profile"
        profiling_data_exists = False
        if profiling_dir.exists():
            profiling_files = list(profiling_dir.glob("*.txt"))
            profiling_data_exists = len(profiling_files) > 0
            logger.info(f"[STIM TEST] Found {len(profiling_files)} profiling files in {profiling_dir}")
        
        self._record_step_time('collect', time.time() - collect_start)
        
        # Collect results
        result = {
            'test_case': test_case_to_dict(test_case),
            'test_dir': str(test_dir),
            'actual_run_dir': str(actual_run_dir),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'profiling_data_exists': profiling_data_exists,
            'videos_deleted_count': 0,  # Will be updated after cleanup
            'success': success
        }
        
        # Save test result
        result_file = test_dir / "test_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _record_step_time(self, step_name: str, duration: float):
        """Record timing for a step and update running average"""
        if step_name not in self._step_timings:
            self._step_timings[step_name] = []
        self._step_timings[step_name].append(duration)
        self._total_steps_completed += 1
    
    def _estimate_remaining_time(self, current_test_idx: int, total_tests: int) -> float:
        """Estimate remaining time based on step averages and protocol well progress"""
        if not self._step_timings and not self._protocol_step_times:
            logger.info(f"[STIM TEST] No timing data yet")
            return 0.0
        
        # Calculate average time per well/step during protocol execution
        avg_protocol_step_time = 0.0
        if self._protocol_step_times:
            avg_protocol_step_time = sum(self._protocol_step_times) / len(self._protocol_step_times)
            logger.info(f"[STIM TEST] Protocol step avg: {avg_protocol_step_time:.2f}s from {len(self._protocol_step_times)} wells")
        
        # Calculate average time for other test steps
        steps_per_test = ['config', 'protocol', 'collect', 'cleanup']
        step_averages = {}
        
        for step in steps_per_test:
            if step in self._step_timings and self._step_timings[step]:
                step_averages[step] = sum(self._step_timings[step]) / len(self._step_timings[step])
                logger.info(f"[STIM TEST] Step '{step}': avg={step_averages[step]:.2f}s, count={len(self._step_timings[step])}")
        
        # Estimate remaining time in current protocol (if in progress)
        time_left_in_current_protocol = 0.0
        if self._protocol_steps_total > 0 and self._protocol_steps_completed < self._protocol_steps_total:
            remaining_wells = self._protocol_steps_total - self._protocol_steps_completed
            if avg_protocol_step_time > 0:
                time_left_in_current_protocol = remaining_wells * avg_protocol_step_time
            else:
                # No well data yet, estimate ~1s per well
                time_left_in_current_protocol = remaining_wells * 1.0
            logger.info(f"[STIM TEST] Current protocol: {self._protocol_steps_completed}/{self._protocol_steps_total} wells done, est remaining: {time_left_in_current_protocol:.1f}s")
        
        # Calculate which test steps are remaining in current test
        current_test_steps_done = 0
        for step in steps_per_test:
            if step in self._step_timings and len(self._step_timings[step]) > current_test_idx:
                current_test_steps_done += 1
        
        # Time remaining for other steps in current test (config, collect, cleanup)
        time_left_in_current_test = time_left_in_current_protocol
        for i, step in enumerate(steps_per_test):
            if i >= current_test_steps_done:  # This step hasn't been done yet
                if step == 'protocol':
                    # Already accounted for above
                    continue
                elif step in step_averages:
                    time_left_in_current_test += step_averages[step]
                else:
                    # Minimal estimate for unknown steps
                    time_left_in_current_test += 0.5
        
        # Calculate time for ALL remaining complete tests
        remaining_tests = total_tests - (current_test_idx + 1)
        if remaining_tests < 0:
            remaining_tests = 0
        
        # For each remaining test, estimate total time
        time_for_remaining_tests = 0.0
        if remaining_tests > 0:
            # Estimate time per complete test
            est_config = step_averages.get('config', 0.5)
            est_protocol = step_averages.get('protocol', 0.0)
            
            # If we have protocol step data, use it; otherwise estimate based on wells
            if est_protocol == 0.0:
                # Estimate protocol time from wells
                if avg_protocol_step_time > 0 and self._protocol_steps_total > 0:
                    est_protocol = avg_protocol_step_time * self._protocol_steps_total
                else:
                    est_protocol = 25.0  # Conservative fallback
            
            est_collect = step_averages.get('collect', 0.5)
            est_cleanup = step_averages.get('cleanup', 0.5)
            
            est_per_test = est_config + est_protocol + est_collect + est_cleanup
            time_for_remaining_tests = est_per_test * remaining_tests
            
            logger.info(f"[STIM TEST] Remaining tests: {remaining_tests}, est per test: {est_per_test:.1f}s, total: {time_for_remaining_tests:.1f}s")
        
        # Total estimate
        estimated_remaining = time_left_in_current_test + time_for_remaining_tests
        
        logger.info(f"[STIM TEST] Time estimate: current_test={current_test_idx+1}/{total_tests}, "
                   f"current_test_remaining={time_left_in_current_test:.1f}s, "
                   f"remaining_tests_time={time_for_remaining_tests:.1f}s, "
                   f"est_remaining={estimated_remaining:.1f}s ({estimated_remaining/60:.1f}m)")
        
        return estimated_remaining
    
    def _apply_test_configuration(self, test_case: StimTestCase, sequenced_capture_executor):
        """Apply test case configuration to the system"""
        from settings_init import settings
        
        # Apply settings that can be changed dynamically
        settings['led_flush_enabled'] = test_case.led_flush_enabled
        
        # Store test case in sequenced_capture_executor for timing strategy selection
        sequenced_capture_executor._current_test_case = test_case
        
        # Reset global latency tracking between tests
        sequenced_capture_executor._avg_led_on_latency = None
        sequenced_capture_executor._avg_led_off_latency = None
        
        logger.info(f"[STIM TEST] Applied configuration: {test_case.name}")
        logger.info(f"[STIM TEST]   Flush={test_case.led_flush_enabled}, "
                   f"ProcessPri={test_case.process_priority_elevated}, "
                   f"ThreadPri={test_case.thread_priority_realtime}")
        logger.info(f"[STIM TEST]   Strategy={test_case.timing_strategy}, "
                   f"Conservative={test_case.use_conservative_off_estimate}")
    
    def _cleanup_video_files(self, test_dir: pathlib.Path) -> int:
        """Delete video files to save space, keep profiling data and images"""
        deleted_count = 0
        
        # Common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        for video_file in test_dir.rglob('*'):
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                try:
                    video_file.unlink()
                    deleted_count += 1
                    logger.debug(f"[STIM TEST] Deleted video: {video_file.name}")
                except Exception as e:
                    logger.warning(f"[STIM TEST] Failed to delete {video_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"[STIM TEST] Deleted {deleted_count} video files from {test_dir.name}")
        
        return deleted_count
    
    def _generate_comparison_report(self) -> pathlib.Path:
        """Generate comprehensive comparison report across all tests"""
        report_file = self.test_run_dir / "comparison_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("STIMULATION TIMING TEST COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Total Tests: {len(self.test_results)}\n")
            f.write(f"Test Suite Directory: {self.test_run_dir}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary table
            f.write("TEST SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'#':<3} {'Test Name':<25} {'Success':<8} {'Duration':<10} {'Profiling':<10}\n")
            f.write("-" * 80 + "\n")
            for idx, result in enumerate(self.test_results):
                test_case = result['test_case']
                success_str = "✓" if result['success'] else "✗"
                duration_str = f"{result['duration_seconds']:.1f}s"
                profiling_str = "Yes" if result['profiling_data_exists'] else "No"
                f.write(f"{idx+1:<3} {test_case['name']:<25} {success_str:<8} {duration_str:<10} {profiling_str:<10}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Detailed results for each test
            for idx, result in enumerate(self.test_results):
                test_case = result['test_case']
                f.write(f"Test {idx + 1}: {test_case['name']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Description: {test_case['description']}\n")
                f.write(f"Duration: {result['duration_seconds']:.1f}s\n")
                f.write(f"Success: {result['success']}\n")
                if not result['success'] and 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                f.write(f"Profiling Data: {'Yes' if result['profiling_data_exists'] else 'No'}\n")
                f.write(f"Videos Deleted: {result['videos_deleted_count']}\n")
                f.write(f"\nConfiguration:\n")
                f.write(f"  Timing Strategy: {test_case['timing_strategy']}\n")
                f.write(f"  LED Flush: {test_case['led_flush_enabled']}\n")
                f.write(f"  Process Priority: {test_case['process_priority_elevated']}\n")
                f.write(f"  Thread Priority: {test_case['thread_priority_realtime']}\n")
                f.write(f"  ON Compensation: {test_case['use_on_latency_compensation']}\n")
                f.write(f"  OFF Compensation: {test_case['use_off_latency_compensation']}\n")
                f.write(f"  Pulse Width Compensation: {test_case['use_pulse_width_compensation']}\n")
                f.write(f"  Conservative OFF Estimate: {test_case['use_conservative_off_estimate']}\n")
                f.write(f"\nTest Directory: {result['test_dir']}\n")
                f.write(f"Actual Run Directory: {result['actual_run_dir']}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("ANALYSIS INSTRUCTIONS\n")
            f.write("=" * 80 + "\n")
            f.write("1. Review stimulation_profile/ folders in each test directory\n")
            f.write("2. Compare 'ACTUAL LED ON TIME' statistics across tests\n")
            f.write("3. Look for tests with:\n")
            f.write("   - Lowest Mean Deviation from target\n")
            f.write("   - Smallest Standard Deviation (consistency)\n")
            f.write("   - Fewest/smallest outliers\n")
            f.write("   - Best frequency accuracy\n")
            f.write("4. The test with best pulse width accuracy should be your default\n")
            f.write("\n")
            f.write("NEXT STEPS:\n")
            f.write("1. Open each test directory and review stimulation_summary.txt\n")
            f.write("2. Identify which configuration provides best timing\n")
            f.write("3. Update default settings to use optimal configuration\n")
        
        logger.info(f"[STIM TEST] Comparison report saved to: {report_file}")
        return report_file
