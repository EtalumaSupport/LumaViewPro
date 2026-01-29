"""
UI Integration for Stimulation Timing Tests

This shows how to integrate the test framework into lumaviewpro.py

ADD TO lumaviewpro.kv:
========================

In the Protocol section, add this button near other protocol controls:

    Button:
        id: run_stim_test_btn
        text: 'Run Stim Test'
        tooltip_text: 'Run systematic stimulation timing optimization tests'
        size_hint_y: None
        height: '35dp'
        on_release: root.run_stimulation_timing_tests()

    Spinner:
        id: stim_test_type_spinner
        text: 'Full Test Suite'
        values: ('Full Test Suite', 'Quick Comparison', 'Current vs Original')
        size_hint_y: None
        height: '30dp'
        tooltip_text: 'Select test suite size'


ADD TO MicroscopeSettings or ProtocolSettings class in lumaviewpro.py:
====================================================================
"""

from modules.stim_timing_test_config import generate_test_cases, get_quick_test_cases
from modules.stim_timing_test_executor import StimTimingTestExecutor
import pathlib


def run_stimulation_timing_tests(self):
    """
    Run systematic stimulation timing optimization tests
    
    This method should be added to the appropriate settings class in lumaviewpro.py
    """
    global lumaview
    
    # Check if protocol is loaded
    if not hasattr(lumaview, 'protocol') or lumaview.protocol is None:
        logger.warning("[STIM TEST] No protocol loaded. Please load a protocol first.")
        # Show popup to user
        return
    
    # Determine which test suite to run
    test_type = self.ids.get('stim_test_type_spinner', None)
    if test_type and test_type.text == 'Quick Comparison':
        test_cases = get_quick_test_cases()
        logger.info("[STIM TEST] Running quick comparison (4 test cases)")
    elif test_type and test_type.text == 'Current vs Original':
        all_cases = generate_test_cases()
        test_cases = [all_cases[0], all_cases[5]]  # Original and Conservative
        logger.info("[STIM TEST] Running current vs original comparison")
    else:
        test_cases = generate_test_cases()
        logger.info(f"[STIM TEST] Running full test suite ({len(test_cases)} test cases)")
    
    # Create test executor
    base_output_dir = pathlib.Path("./capture/stim_timing_tests")
    test_executor = StimTimingTestExecutor(
        protocol_executor=lumaview.scope.executor,
        base_output_dir=base_output_dir
    )
    
    # Set up callbacks for UI updates
    def on_test_start(test_case, current, total):
        logger.info(f"[STIM TEST UI] Starting test {current}/{total}: {test_case.name}")
        # Update UI progress bar or status text
        # self.ids.stim_test_status.text = f"Running test {current}/{total}: {test_case.name}"
    
    def on_test_complete(test_case, result):
        logger.info(f"[STIM TEST UI] Completed test: {test_case.name}")
        # Update UI
    
    def on_all_tests_complete(results):
        logger.info(f"[STIM TEST UI] All tests complete! Total: {len(results)}")
        # Show completion dialog
        # Open file explorer to results directory
        # self.show_test_results_dialog(test_executor.test_run_dir)
    
    test_executor.on_test_start = on_test_start
    test_executor.on_test_complete = on_test_complete
    test_executor.on_all_tests_complete = on_all_tests_complete
    
    # Get protocol parameters
    protocol_params = {
        'max_scans': 1,  # Run each test for 1 full scan
        'parent_dir': None,  # Will be set by test executor
        'run_trigger_source': 'stim_test'
    }
    
    # Disable UI controls during test
    # self.ids.run_stim_test_btn.disabled = True
    
    # Run tests in background thread to avoid blocking UI
    import threading
    test_thread = threading.Thread(
        target=test_executor.start_test_suite,
        args=(test_cases, lumaview.protocol, protocol_params),
        daemon=False
    )
    test_thread.start()
    
    logger.info("[STIM TEST] Test suite started in background")


# Example of how to show results dialog
def show_test_results_dialog(test_dir: pathlib.Path):
    """Show dialog with test results and options to view/analyze"""
    import subprocess
    
    # Open file explorer to results directory
    if sys.platform.startswith('win'):
        subprocess.Popen(['explorer', str(test_dir)])
    elif sys.platform.startswith('darwin'):
        subprocess.Popen(['open', str(test_dir)])
    else:
        subprocess.Popen(['xdg-open', str(test_dir)])
    
    logger.info(f"[STIM TEST] Opened results directory: {test_dir}")


"""
USAGE INSTRUCTIONS:
==================

1. Add the UI elements from above to lumaviewpro.kv

2. Add this import to lumaviewpro.py:
   from modules.stim_timing_test_config import generate_test_cases, get_quick_test_cases
   from modules.stim_timing_test_executor import StimTimingTestExecutor

3. Add the run_stimulation_timing_tests method to your settings class

4. Load a protocol with stimulation enabled

5. Click "Run Stim Test" button

6. Select test suite type (Full/Quick/Current vs Original)

7. Wait for tests to complete (will take time based on protocol duration × number of tests)

8. Review results in ./capture/stim_timing_tests/stim_timing_tests_[timestamp]/
   - Each test has its own directory with profiling data
   - comparison_report.txt shows overview
   - Videos are automatically deleted to save space


ANALYZING RESULTS:
==================

Compare these metrics across tests:

1. ACTUAL LED ON TIME:
   - Mean should be close to target pulse width
   - Std Dev should be low (consistent)
   - Max should never exceed target significantly
   - Outliers should be minimal

2. FREQUENCY ACCURACY:
   - Actual frequency should match target
   - Period std dev should be low

3. Command Latencies:
   - Lower and more consistent is better
   - Outliers indicate OS scheduling issues

Best test will have:
- Accurate pulse width (mean ≈ target, low deviation)
- No dangerous overshoots (max ≤ target + 2ms)
- Consistent timing (low std dev)
- Good frequency accuracy
"""
