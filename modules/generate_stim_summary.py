"""
Standalone script to generate stimulation timing summary from profile files.

Usage:
    python generate_stim_summary.py <folder_path>
    
    folder_path: Path to run directory (containing stimulation_profile/) 
                 or direct path to stimulation_profile/ folder
"""

import sys
import datetime
import numpy as np
from pathlib import Path


def parse_stimulation_profile(filepath):
    """Parse a stimulation profile file and extract timing data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata
        data = {'filename': filepath.name}
        
        lines = content.split('\n')
        for line in lines:
            if 'Color:' in line or 'Stimulation Profiling Report -' in line:
                if '-' in line:
                    data['color'] = line.split('-')[-1].strip()
            elif 'Frequency:' in line:
                data['frequency'] = line.split(':')[1].split('Hz')[0].strip()
            elif 'Pulse Width:' in line:
                data['pulse_width'] = line.split(':')[1].split('ms')[0].strip()
            elif 'Pulses Executed:' in line:
                data['pulses_executed'] = int(line.split(':')[1].strip())
            elif 'End Reason:' in line:
                data['end_reason'] = line.split(':')[1].strip()
        
        # Extract timing data
        data['actual_on'] = []
        data['led_on_cmd'] = []
        data['led_off_cmd'] = []
        
        # Parse timing sections
        in_actual_on = False
        in_led_on = False
        in_led_off = False
        
        for line in lines:
            if 'LED ACTUAL ON-TIME' in line:
                in_actual_on = True
                in_led_on = False
                in_led_off = False
            elif 'LED ON TIMINGS' in line and 'ACTUAL' not in line:
                in_led_on = True
                in_actual_on = False
                in_led_off = False
            elif 'LED OFF TIMINGS' in line:
                in_led_off = True
                in_actual_on = False
                in_led_on = False
            elif line.startswith('  Pulse '):
                # Extract duration from pulse lines
                if 'Duration=' in line:
                    duration_str = line.split('Duration=')[1].split('ms')[0].strip()
                    try:
                        duration = float(duration_str)
                        if in_actual_on:
                            data['actual_on'].append(duration)
                        elif in_led_on:
                            data['led_on_cmd'].append(duration)
                        elif in_led_off:
                            data['led_off_cmd'].append(duration)
                    except ValueError:
                        pass
        
        return data
        
    except Exception as e:
        print(f"ERROR: Failed to parse {filepath}: {e}")
        return None


def write_timing_stats(f, values):
    """Write statistical summary for timing values."""
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    median = np.median(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    
    f.write(f"  Count:      {len(values)}\n")
    f.write(f"  Mean:       {mean:.6f} ms\n")
    f.write(f"  Median:     {median:.6f} ms\n")
    f.write(f"  Std Dev:    {std:.6f} ms\n")
    f.write(f"  Min:        {min_val:.6f} ms\n")
    f.write(f"  Max:        {max_val:.6f} ms\n")
    f.write(f"  95th %ile:  {p95:.6f} ms\n")
    f.write(f"  99th %ile:  {p99:.6f} ms\n")


def write_outlier_details(f, values, sources, color, expected_values=None):
    """Write detailed outlier information with source files."""
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    
    # 3-sigma outliers
    threshold_3sigma = mean + 3 * std
    outliers_3sigma = [(i, v, sources[i]) for i, v in enumerate(values) if v > threshold_3sigma]
    
    f.write(f"  3-Sigma Threshold: {threshold_3sigma:.6f} ms\n")
    f.write(f"  3-Sigma Outliers: {len(outliers_3sigma)} ({len(outliers_3sigma)/len(values)*100:.2f}%)\n")
    
    if outliers_3sigma:
        f.write(f"\n  3-Sigma Outlier Details (showing all):\n")
        for idx, value, source in sorted(outliers_3sigma, key=lambda x: x[1], reverse=True):
            f.write(f"    {value:.6f} ms - Pulse #{source['pulse_num']} in {source['file']}\n")
    
    # For actual on-time, show deviations from expected pulse width
    if expected_values is not None and len(expected_values) == len(values):
        expected_arr = np.array(expected_values)
        deviations = arr - expected_arr
        # Pulses that were >3ms over the intended LED on time
        over_3ms = [(i, v, sources[i], deviations[i]) for i, v in enumerate(values) if deviations[i] > 3.0]
        
        f.write(f"\n  Pulses >3ms Over Intended: {len(over_3ms)} ({len(over_3ms)/len(values)*100:.2f}%)\n")
        
        if over_3ms:
            f.write(f"\n  Details (showing all pulses >3ms over intended):\n")
            for idx, value, source, deviation in sorted(over_3ms, key=lambda x: x[3], reverse=True):
                expected = expected_values[idx]
                f.write(f"    {value:.6f} ms (expected {expected:.3f} ms, +{deviation:.3f} ms deviation) - Pulse #{source['pulse_num']} in {source['file']}\n")


def generate_stimulation_summary(folder_path):
    """Generate a summary report aggregating all stimulation profiling data."""
    folder = Path(folder_path)
    
    # Check if this is the run directory or the profile directory
    if folder.name == "stimulation_profile":
        profile_dir = folder
        run_dir = folder.parent
    else:
        profile_dir = folder / "stimulation_profile"
        run_dir = folder
    
    if not profile_dir.exists():
        print(f"ERROR: Stimulation profile directory not found: {profile_dir}")
        return False
    
    # Find all profile files
    profile_files = list(profile_dir.glob("stimulation_profile_*.txt"))
    if not profile_files:
        print(f"ERROR: No stimulation profile files found in {profile_dir}")
        return False
    
    print(f"Found {len(profile_files)} profile files")
    
    try:
        # Data structures to aggregate - separated by color
        all_data_by_color = {}  # color -> timing_type -> {'values': [], 'sources': []}
        
        file_summaries = []
        colors_found = set()
        
        # Parse each profile file
        for profile_file in sorted(profile_files):
            print(f"  Parsing: {profile_file.name}")
            file_data = parse_stimulation_profile(profile_file)
            if file_data:
                file_summaries.append(file_data)
                color = file_data.get('color', 'Unknown')
                colors_found.add(color)
                
                # Initialize color data structure if needed
                if color not in all_data_by_color:
                    all_data_by_color[color] = {
                        'actual_on': {'values': [], 'sources': [], 'pulse_widths': []},
                        'led_on_cmd': {'values': [], 'sources': []},
                        'led_off_cmd': {'values': [], 'sources': []}
                    }
                
                # Aggregate data with source tracking, separated by color
                for timing_type in ['actual_on', 'led_on_cmd', 'led_off_cmd']:
                    if timing_type in file_data and file_data[timing_type]:
                        for idx, value in enumerate(file_data[timing_type]):
                            all_data_by_color[color][timing_type]['values'].append(value)
                            all_data_by_color[color][timing_type]['sources'].append({
                                'file': profile_file.name,
                                'pulse_num': idx
                            })
                
                # Track expected pulse width for actual_on validation
                if 'pulse_width' in file_data and file_data['pulse_width'] != 'N/A':
                    try:
                        pw = float(file_data['pulse_width'])
                        all_data_by_color[color]['actual_on']['pulse_widths'].extend([pw] * len(file_data.get('actual_on', [])))
                    except:
                        pass
        
        # Generate summary report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = profile_dir / f"stimulation_summary_{timestamp}.txt"
        
        print(f"\nGenerating summary: {summary_file}")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STIMULATION PROFILING SUMMARY - ENTIRE RUN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Run Directory: {run_dir}\n")
            f.write(f"Number of Profile Files: {len(profile_files)}\n")
            f.write(f"Colors Found: {', '.join(sorted(colors_found))}\n")
            
            # Calculate total pulses across all colors
            total_pulses = sum(len(all_data_by_color[c]['actual_on']['values']) for c in all_data_by_color)
            f.write(f"Total Pulses Analyzed: {total_pulses}\n")
            f.write("\n")
            
            # Statistics separated by color/channel
            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICS BY COLOR/CHANNEL\n")
            f.write("=" * 80 + "\n\n")
            
            for color in sorted(colors_found):
                if color not in all_data_by_color:
                    continue
                
                color_data = all_data_by_color[color]
                num_pulses = len(color_data['actual_on']['values'])
                
                f.write("=" * 80 + "\n")
                f.write(f"COLOR: {color}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total Pulses: {num_pulses}\n\n")
                
                # LED Actual On-Time
                if color_data['actual_on']['values']:
                    f.write("LED ACTUAL ON-TIME (From LED ON return to LED OFF call):\n")
                    f.write("-" * 80 + "\n")
                    write_timing_stats(f, color_data['actual_on']['values'])
                    
                    # Show expected pulse width if available
                    if color_data['actual_on']['pulse_widths']:
                        expected_pw = np.mean(color_data['actual_on']['pulse_widths'])
                        actual_mean = np.mean(color_data['actual_on']['values'])
                        deviation = actual_mean - expected_pw
                        f.write(f"  Expected PW: {expected_pw:.3f} ms\n")
                        f.write(f"  Deviation:   {deviation:.6f} ms ({deviation/expected_pw*100:.2f}%)\n")
                    f.write("\n")
                
                # LED ON Command Duration
                if color_data['led_on_cmd']['values']:
                    f.write("LED ON COMMAND DURATION:\n")
                    f.write("-" * 80 + "\n")
                    write_timing_stats(f, color_data['led_on_cmd']['values'])
                    f.write("\n")
                
                # LED OFF Command Duration
                if color_data['led_off_cmd']['values']:
                    f.write("LED OFF COMMAND DURATION:\n")
                    f.write("-" * 80 + "\n")
                    write_timing_stats(f, color_data['led_off_cmd']['values'])
                    f.write("\n")
            
            # Outlier Analysis by color
            f.write("\n" + "=" * 80 + "\n")
            f.write("OUTLIER ANALYSIS BY COLOR\n")
            f.write("=" * 80 + "\n\n")
            
            for color in sorted(colors_found):
                if color not in all_data_by_color:
                    continue
                
                color_data = all_data_by_color[color]
                
                f.write("=" * 80 + "\n")
                f.write(f"COLOR: {color}\n")
                f.write("=" * 80 + "\n\n")
                
                for timing_name, timing_label in [
                    ('actual_on', 'LED Actual On-Time'),
                    ('led_on_cmd', 'LED ON Command'),
                    ('led_off_cmd', 'LED OFF Command')
                ]:
                    if color_data[timing_name]['values']:
                        f.write(f"{timing_label} Outliers:\n")
                        f.write("-" * 80 + "\n")
                        # For actual_on, pass expected pulse widths for deviation analysis
                        expected_vals = color_data['actual_on']['pulse_widths'] if timing_name == 'actual_on' else None
                        write_outlier_details(
                            f, 
                            color_data[timing_name]['values'],
                            color_data[timing_name]['sources'],
                            color,
                            expected_vals
                        )
                        f.write("\n")
        
        print(f"\n✓ Summary report saved to: {summary_file}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to generate stimulation summary: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("\nERROR: Please provide a folder path")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not Path(folder_path).exists():
        print(f"ERROR: Folder does not exist: {folder_path}")
        sys.exit(1)
    
    success = generate_stimulation_summary(folder_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
