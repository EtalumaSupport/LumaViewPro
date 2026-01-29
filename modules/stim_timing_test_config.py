"""
Stimulation Timing Test Configuration
Defines test cases for systematically evaluating LED timing optimizations
"""

import dataclasses
from typing import List, Dict, Any
import itertools


@dataclasses.dataclass
class StimTestCase:
    """Configuration for a single stimulation timing test"""
    name: str
    description: str
    
    # Timing strategy settings
    timing_strategy: str  # 'original', 'basic', 'predictive', 'conservative'
    led_flush_enabled: bool
    process_priority_elevated: bool
    thread_priority_realtime: bool
    
    # Compensation settings
    use_on_latency_compensation: bool
    use_off_latency_compensation: bool
    use_pulse_width_compensation: bool
    use_conservative_off_estimate: bool  # 2x average vs 1x average
    
    # Test metadata
    test_id: int = 0


def generate_test_cases() -> List[StimTestCase]:
    """Generate comprehensive test case matrix"""
    
    test_cases = []
    test_id = 0
    
    # Test 1: Original implementation (baseline)
    test_cases.append(StimTestCase(
        name="01_Original_Baseline",
        description="Original implementation with no optimizations",
        timing_strategy='original',
        led_flush_enabled=False,
        process_priority_elevated=False,
        thread_priority_realtime=False,
        use_on_latency_compensation=False,
        use_off_latency_compensation=False,
        use_pulse_width_compensation=False,
        use_conservative_off_estimate=False,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 2: Original + Flush only
    test_cases.append(StimTestCase(
        name="02_Original_With_Flush",
        description="Original implementation with flush enabled",
        timing_strategy='original',
        led_flush_enabled=True,
        process_priority_elevated=False,
        thread_priority_realtime=False,
        use_on_latency_compensation=False,
        use_off_latency_compensation=False,
        use_pulse_width_compensation=False,
        use_conservative_off_estimate=False,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 3: Original + Priorities only
    test_cases.append(StimTestCase(
        name="03_Original_With_Priorities",
        description="Original implementation with process and thread priorities",
        timing_strategy='original',
        led_flush_enabled=False,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=False,
        use_off_latency_compensation=False,
        use_pulse_width_compensation=False,
        use_conservative_off_estimate=False,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 4: Original + Flush + Priorities
    test_cases.append(StimTestCase(
        name="04_Original_Flush_Priorities",
        description="Original implementation with flush and priorities",
        timing_strategy='original',
        led_flush_enabled=True,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=False,
        use_off_latency_compensation=False,
        use_pulse_width_compensation=False,
        use_conservative_off_estimate=False,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 5: Basic predictive compensation
    test_cases.append(StimTestCase(
        name="05_Basic_Predictive",
        description="Predictive compensation with 1x average latency",
        timing_strategy='predictive',
        led_flush_enabled=True,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=True,
        use_off_latency_compensation=True,
        use_pulse_width_compensation=True,
        use_conservative_off_estimate=False,  # 1x average
        test_id=test_id
    ))
    test_id += 1
    
    # Test 6: Conservative predictive compensation (CURRENT BEST)
    test_cases.append(StimTestCase(
        name="06_Conservative_Predictive",
        description="Predictive compensation with 2x conservative OFF estimate",
        timing_strategy='conservative',
        led_flush_enabled=True,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=True,
        use_off_latency_compensation=True,
        use_pulse_width_compensation=True,
        use_conservative_off_estimate=True,  # 2x average (current implementation)
        test_id=test_id
    ))
    test_id += 1
    
    # Test 7: Conservative without flush (test flush impact)
    test_cases.append(StimTestCase(
        name="07_Conservative_No_Flush",
        description="Conservative compensation but without flush",
        timing_strategy='conservative',
        led_flush_enabled=False,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=True,
        use_off_latency_compensation=True,
        use_pulse_width_compensation=True,
        use_conservative_off_estimate=True,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 8: Conservative without priorities (test priority impact)
    test_cases.append(StimTestCase(
        name="08_Conservative_No_Priorities",
        description="Conservative compensation but without elevated priorities",
        timing_strategy='conservative',
        led_flush_enabled=True,
        process_priority_elevated=False,
        thread_priority_realtime=False,
        use_on_latency_compensation=True,
        use_off_latency_compensation=True,
        use_pulse_width_compensation=True,
        use_conservative_off_estimate=True,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 9: Only ON compensation (isolate ON impact)
    test_cases.append(StimTestCase(
        name="09_Only_ON_Compensation",
        description="Only ON latency compensation, no OFF compensation",
        timing_strategy='basic',
        led_flush_enabled=True,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=True,
        use_off_latency_compensation=False,
        use_pulse_width_compensation=False,
        use_conservative_off_estimate=False,
        test_id=test_id
    ))
    test_id += 1
    
    # Test 10: Only OFF compensation (isolate OFF impact)
    test_cases.append(StimTestCase(
        name="10_Only_OFF_Compensation",
        description="Only OFF latency compensation, no ON compensation",
        timing_strategy='basic',
        led_flush_enabled=True,
        process_priority_elevated=True,
        thread_priority_realtime=True,
        use_on_latency_compensation=False,
        use_off_latency_compensation=True,
        use_pulse_width_compensation=True,
        use_conservative_off_estimate=True,
        test_id=test_id
    ))
    test_id += 1
    
    return test_cases


def get_quick_test_cases() -> List[StimTestCase]:
    """Get a smaller subset of critical test cases for quick comparison"""
    all_cases = generate_test_cases()
    # Return: Original, Original+Flush+Priority, Basic Predictive, Conservative (current)
    return [all_cases[0], all_cases[3], all_cases[4], all_cases[5]]


def test_case_to_dict(test_case: StimTestCase) -> Dict[str, Any]:
    """Convert test case to dictionary for JSON serialization"""
    return dataclasses.asdict(test_case)


def get_test_matrix_summary() -> str:
    """Get human-readable summary of test matrix"""
    cases = generate_test_cases()
    summary = "STIMULATION TIMING TEST MATRIX\n"
    summary += "=" * 80 + "\n\n"
    
    for case in cases:
        summary += f"Test {case.test_id + 1}: {case.name}\n"
        summary += f"  Description: {case.description}\n"
        summary += f"  Strategy: {case.timing_strategy}\n"
        summary += f"  Flush: {case.led_flush_enabled}, "
        summary += f"Process Priority: {case.process_priority_elevated}, "
        summary += f"Thread Priority: {case.thread_priority_realtime}\n"
        summary += f"  Compensations: ON={case.use_on_latency_compensation}, "
        summary += f"OFF={case.use_off_latency_compensation}, "
        summary += f"PulseWidth={case.use_pulse_width_compensation}, "
        summary += f"Conservative={case.use_conservative_off_estimate}\n"
        summary += "\n"
    
    return summary


if __name__ == "__main__":
    # Print test matrix when run directly
    print(get_test_matrix_summary())
    print(f"\nTotal test cases: {len(generate_test_cases())}")
    print(f"Quick test cases: {len(get_quick_test_cases())}")
