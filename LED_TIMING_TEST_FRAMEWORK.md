# LED Timing Test Framework - User Guide

## Overview
The LED timing test framework provides systematic testing of different LED stimulation timing optimizations to identify the best configuration for minimizing pulse width errors and preventing phototoxicity.

## Location in UI
The test controls are located in the **Protocol Settings** section:
- **Test Type Spinner**: Choose between "All Tests (10)" or "Quick Tests (4)"
- **Run LED Timing Tests Button**: Launches the test suite

## Prerequisites
1. Load a protocol with LED stimulation enabled on at least one step
2. Ensure the protocol has been validated and runs successfully
3. Free up disk space (tests will temporarily store videos before deletion)

## How It Works

### Test Execution
1. The framework runs your loaded protocol multiple times (10 or 4 times depending on selection)
2. Each test uses a different combination of timing optimizations:
   - LED flush (on/off)
   - Process priority elevation (on/off)
   - Thread priority (realtime/normal)
   - Latency compensation strategies (various combinations)
   - Conservative OFF estimation (on/off)

3. For each test:
   - Applies the specific configuration
   - Runs ONE SCAN of your protocol (not full duration)
   - Collects detailed timing profiling data
   - Deletes video files to save space (keeps profiling data)
   - Moves to next test after brief delay

### Output Structure
Tests are saved to: `{capture_dir}/stim_timing_tests_{timestamp}/`

Each test creates a subdirectory:
```
test_00_original/
  - test_case_config.json         (configuration used)
  - test_result.json               (success/duration/etc)
  - stimulation_profile/           (timing data per color)
    - stimulation_profile_red_*.txt
    - stimulation_summary.txt
```

### Analysis Report
After all tests complete, a `comparison_report.txt` is generated with:
- Summary table of all tests
- Detailed configuration for each test
- Instructions for analyzing results

## Test Configurations

### All Tests (10 configurations)
1. **original**: No optimizations (baseline)
2. **flush_only**: Just LED flush enabled
3. **priorities_only**: Just process+thread priority elevation
4. **flush_and_priorities**: Both flush and priorities
5. **basic_predictive**: Full compensations without conservative OFF
6. **conservative_current**: Full compensations WITH conservative OFF (current default)
7. **no_pulse_compensation**: All except pulse width compensation
8. **no_conservative_off**: All except conservative OFF estimate
9. **no_on_compensation**: All except ON latency compensation
10. **no_off_compensation**: All except OFF latency compensation

### Quick Tests (4 configurations)
1. **original**: Baseline
2. **flush_only**: Just flush
3. **basic_predictive**: Full predictive
4. **conservative_current**: Current default (most conservative)

## Analyzing Results

### Step 1: Review Individual Tests
For each test directory, examine `stimulation_profile/stimulation_summary.txt`:
```
ACTUAL LED ON TIME
  - Mean: X.XX ms (deviation from target)
  - Std Dev: X.XX ms (consistency)
  - Min/Max: X.XX ms / X.XX ms
  - Outliers: X pulses beyond 3σ
```

### Step 2: Compare Metrics Across Tests
Look for the test with:
- **Lowest mean deviation** from target pulse width (closest to 0)
- **Smallest standard deviation** (most consistent)
- **Fewest outliers** (most reliable)
- **Best frequency accuracy** (actual vs target)

### Step 3: Prioritize Pulse Width Accuracy
⚠️ **CRITICAL**: Pulse width accuracy is MORE important than frequency accuracy!
- Long pulses → phototoxicity (cell damage)
- Missed frequency → less critical than overexposure

### Step 4: Update Default Configuration
Once you identify the optimal test, update your system to use those settings permanently.

## Important Notes

### Videos Are Automatically Deleted
- Videos are recorded during each test (to keep variables constant)
- After each test completes, videos are automatically deleted
- Only profiling data and images are kept
- This saves disk space while maintaining consistent test conditions

### Test Duration
- **Quick Tests**: ~8-15 minutes (4 × protocol scan time + overhead)
- **All Tests**: ~20-35 minutes (10 × protocol scan time + overhead)

### What If Tests Fail?
If a test fails to complete:
- Check the test_result.json for error information
- Review logs for error messages
- Verify protocol runs successfully outside test framework
- Ensure adequate disk space

### Interpreting Conservative OFF Estimates
The "conservative_current" test (default) uses 2× average OFF latency:
- Protects against OFF latency outliers
- Prevents LED staying on too long
- May result in slightly shorter pulses
- Prioritizes safety over exact pulse width

## Example Workflow

1. Load your protocol with LED stimulation
2. Select "Quick Tests (4)" for initial comparison
3. Click "Run LED Timing Tests"
4. Wait for completion (~10-15 minutes)
5. When prompted, open the results folder
6. Review `comparison_report.txt`
7. Examine each test's `stimulation_summary.txt`
8. Identify test with best pulse width accuracy
9. If needed, run "All Tests (10)" to fine-tune
10. Update default settings based on best configuration

## Technical Details

### Timing Measurement
- Uses `time.perf_counter()` for sub-millisecond accuracy
- Measures command start, end, and duration
- Tracks global running averages across all colors/pulses
- Records absolute timestamps for temporal analysis

### Compensation Strategies
- **ON Compensation**: Fire ON command early to account for latency
- **OFF Compensation**: Fire OFF command early to prevent overexposure
- **Pulse Width Compensation**: Adjust pulse width for OFF latency
- **Conservative OFF**: Use 2× average OFF latency for safety

### Priority Elevation
- **Process Priority**: HIGH_PRIORITY_CLASS during protocol
- **Thread Priority**: THREAD_PRIORITY_TIME_CRITICAL for stimulation
- Reduces OS scheduling jitter
- May conflict with other realtime applications

## Troubleshooting

### "No LED Stimulation" Error
- Enable LED stimulation on at least one protocol step
- Check that stimulation parameters are valid (frequency > 0, etc.)

### Test Hangs or Times Out
- Each test has 5-minute timeout
- Check if protocol can run manually
- Review logs for hardware communication issues

### Out of Disk Space
- Tests automatically delete videos between tests
- Ensure at least 10GB free for test run
- Profiling data is small (few MB per test)

### Inconsistent Results
- Environmental factors (CPU load) can affect timing
- Run tests when system is idle
- Consider running multiple test suites and averaging
- Check for background tasks during testing

## Support
For questions or issues with the test framework, check logs at:
`logs/LVP_Log/` for detailed error messages and timing diagnostics.
