"""
-------------------------------------------
🚦 FINAL CALIBRATED DELAY CALCULATION LOGIC
-------------------------------------------

For each sensor, the final calibrated delay is computed through a multi-stage process:

1. 📡 Onsite Delay Estimation:
   - For each detected anomaly, calculate:
        delay = stabilization_time - start_time
   - Apply bounds: discard negative delays or delays > `delay_acceptance_limit`.
   - Remove outliers using IQR filtering.
   - Average the remaining delays → `onsite_delay`.

2. 🧪 Blend Lab + Onsite Delays:
   - final_delay = lab_weight * lab_delay + onsite_weight * onsite_delay
   - Default weights: 0.5 each (from config["lab_onsite_weight"]).
   - Result is saved as `lab+onsite_delays`.

3. 🧭 Reference Sensor Adjustment (Optional):
   - If config["use_reference_sensor"] is True and a reference_sensor_id is set,
     the delay is adjusted relative to the reference sensor delay using:
        apply_reference_strategy(final_delay, ref_delay)
   - Result is still part of `final_calibrated_delays_with_reference_sensor`.

4. 🧑‍🏫 Human-In-The-Loop Adjustment (Optional):
   - If human-reviewed delays are available, they are blended (not overridden):
        final_delay = 0.5 * final_delay + 0.5 * human_adjusted_delay
   - This ensures robustness and incorporates manual insight.
   - Final result is saved as both:
        - "final_calibrated_delays_with_reference_sensor"
        - "human_adjusted_delays" (raw average)

🔍 How to Know if Reference Sensor Was Used:
   - Check config["use_reference_sensor"] == True
   - And config["reference_sensor_id"] is defined
   - If these are both true, then reference alignment is applied.
"""


import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.widgets import Cursor
from scipy.signal import correlate
import numpy as np
from collections import defaultdict


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorDelayDetector:
    # def __init__(self, config_path: str = None, delay_file_path: str = 'baseline_delays.json', debug: bool = False):
    def __init__(
        self,
        config_path: str = None,
        delay_file_path: str = 'baseline_delays.json',
        debug: bool = False,
        sensor_metadata: Optional[Dict[str, str]] = None
    ):
        self.debug = debug
        self.config = {
            "use_reference_sensor": False,
            "reference_sensor_id": None,
            "reference_strategy": "median_consensus", # Options: "median_consensus", "fast_response"
            "stabilization_window": 5,
            "stabilization_threshold": 0.05,
            "minimum_anomaly_magnitude": 1.5,
            "confidence_threshold": 0.50,
            "delay_acceptance_limit": 60,
            "lab_onsite_weight": [0.7, 0.3],
            "suppress_event_output": True,
            "use_human_delay": False,
            "human_review_figures": False,
            "num_human_review_samples": 1,
            "only_uncertain_events": True,
            "human_review_window_sec": 120
        }

        # Auto-enable reference sensor if any IRS sensor is found
        if sensor_metadata:
            for sensor_id, sensor_type in sensor_metadata.items():
                if sensor_type.upper() == 'IRS':
                    self.config["use_reference_sensor"] = True
                    self.config["reference_sensor_id"] = sensor_id
                    logger.info(f"Auto-enabled reference sensor: {sensor_id}")
                    break
        
        self.sensor_metadata = sensor_metadata

        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

        self.baseline_delays = {}
        try:
            with open(delay_file_path, 'r') as f:
                self.baseline_delays = json.load(f)
            logger.info(f"Baseline delays loaded from {delay_file_path}. Found delays for {len(self.baseline_delays)} sensors.")
        except FileNotFoundError:
            logger.warning(f"{delay_file_path} not found. Will rely on real-time estimation.")
        except Exception as e:
            logger.error(f"Error loading baseline delays from {delay_file_path}: {e}")

        self.detected_anomalies = defaultdict(list)
        self.sensor_delays = {}
        self.calibrated_delays = {}
        self.human_adjusted_delays = {}
        self.onsite_delays = {}
        self.unadjusted_delays = {}
        self.onsite_human_delays = {}

        # print(f"[DEBUG] DelayDetector config loaded: {json.dumps(self.config, indent=2)}")

    def detect_anomalies(
    self,
    sensor_data: Dict[str, pd.DataFrame],
    threshold_dict: Dict[str, float] = None,
    model=None,
    scaler=None,
    sensitivity: str = "medium") -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies for delay calibration.
        - Uses AdaptiveUnsupervisedAutoencoder.predict_anomalies_home_optimized
        for consistency with training.
        - Still extracts per-sensor reconstruction errors to build rich event
        dictionaries (start, threshold crossing, peak, confidence).
        - Output format matches the original method so calculate_delays works.
        """
        anomalies = defaultdict(list)

        # print("\n[DEBUG] detect_anomalies using HOME-OPTIMIZED detector")
        print(f"[DEBUG] Sensitivity: {sensitivity}")

        # === Prepare aligned input ===
        sensor_ids = list(sensor_data.keys())
        timestamps = sensor_data[sensor_ids[0]]["timestamp"].reset_index(drop=True)
        X = np.stack([sensor_data[s]["value"].reset_index(drop=True) for s in sensor_ids], axis=1)

        if not hasattr(model, "predict_anomalies_home_optimized"):
            raise RuntimeError("Loaded model does not support predict_anomalies_home_optimized")

        # === Run detection (home-optimized) ===
        # This method runs the full anomaly detection pipeline:
        #   - Uses the autoencoder reconstruction error
        #   - Combines it with Isolation Forest and Elliptic Envelope
        #   - Applies sensitivity rules (low/medium/high)
        #   - Returns a global anomaly flag per sample (event-level decision)
        anomaly_flags, _, drift_info = model.predict_anomalies_home_optimized(
            X, adapt=False, sensitivity=sensitivity, verbose=False
        )

        print(f"[DEBUG] Model thresholds per sensor:")
        for sid in sensor_ids:
            t = model.sensor_thresholds.get(sid, None)
            print(f"   - {sid}: {t}")
        # print(f"[DEBUG] Drift info: {drift_info}")

        print(f"[DEBUG] Detection finished | Total anomalies: {np.sum(anomaly_flags)} / {len(anomaly_flags)}")
        # print(f"[DEBUG] Drift info: {drift_info}")
        
        # === Get per-sensor reconstruction errors (for delay + confidence) ===
        # Here we bypass the consensus logic and use ONLY the autoencoder’s reconstruction:
        #   - Needed because delay calibration requires sensor-specific errors,
        #     not just the global anomaly flag.
        #   - Lets us extract event boundaries (start, crossing, peak) and confidence
        #     on a per-sensor basis.
        X_scaled = model.scaler.transform(X)
        recon = model.autoencoder.predict(X_scaled)
        if recon.ndim == 1:
            recon = recon.reshape(-1, X_scaled.shape[1])
        elif recon.shape[1] != X_scaled.shape[1]:
            recon = recon.reshape(-1, X_scaled.shape[1])
        recon_errors = (X_scaled - recon) ** 2

        # Margin factor used to normalize confidence values:
        #   confidence = (peak_value - threshold) / (threshold * margin)
        # With margin=0.5:
        #   - A peak that is 50% above threshold → confidence ~ 1.0
        #   - Values closer to threshold scale down proportionally
        margin = 0.5  # for confidence scaling

        # === Build event dictionaries per sensor ===
        for i, s in enumerate(sensor_ids):
            error_series = recon_errors[:, i]

            if threshold_dict and s in threshold_dict:
                threshold = threshold_dict[s]
            elif hasattr(model, "sensor_thresholds") and s in model.sensor_thresholds:
                threshold = model.sensor_thresholds[s]
            else:
                threshold = model.sensor_thresholds.get(s, np.percentile(error_series, 95))

            print(f"[DEBUG] Sensor {s}: Using threshold {threshold} for event extraction")

            # sensor-specific anomaly flags (not just global)
            anomaly_flags_sensor = (error_series > threshold).astype(int)
            idx = np.where(anomaly_flags_sensor == 1)[0]
            if len(idx) == 0:
                continue

            # print(f"[DEBUG] {s}: {len(idx)} anomaly points out of {len(error_series)} total")
            if len(idx) > 1:
                gaps = np.diff(idx)  # Calculate gaps between anomaly points
                large_gaps = gaps[gaps > 5]
                # print(f"[DEBUG] {s}: {len(large_gaps)} gaps > 5 samples, max gap: {gaps.max()}")
                # print(f"[DEBUG] {s}: Gap statistics - mean: {gaps.mean():.1f}, median: {np.median(gaps):.1f}")     
                           
            # === Group contiguous indices into eve
            events = []
            current_start = idx[0]

            for j in range(1, len(idx)):
                if idx[j] - idx[j - 1] > 5:  # gap => new event
                    current_end = idx[j - 1]

                    # find start + threshold crossing
                    anomaly_start_idx, threshold_crossing_idx = self._find_anomaly_boundaries(
                        error_series, threshold, current_start, current_end
                    )
                    peak_idx = current_start + np.argmax(error_series[current_start:current_end + 1])

                    peak_value = error_series[peak_idx]
                    confidence = float(min(max((peak_value - threshold) / (threshold * margin), 0.0), 1.0))

                    events.append({
                        "anomaly_start_idx": int(anomaly_start_idx),
                        "threshold_crossing_idx": int(threshold_crossing_idx),
                        "anomaly_start_time": pd.Timestamp(timestamps[anomaly_start_idx]),
                        "threshold_crossing_time": pd.Timestamp(timestamps[threshold_crossing_idx]),
                        "peak_idx": int(peak_idx),
                        "peak_time": pd.Timestamp(timestamps[peak_idx]),
                        "end_idx": int(current_end),
                        "end_time": pd.Timestamp(timestamps[current_end]),
                        "confidence": confidence
                    })

                    current_start = idx[j]

            # handle last event
            current_end = idx[-1]
            anomaly_start_idx, threshold_crossing_idx = self._find_anomaly_boundaries(
                error_series, threshold, current_start, current_end
            )
            peak_idx = current_start + np.argmax(error_series[current_start:current_end + 1])

            peak_value = error_series[peak_idx]
            confidence = float(min(max((peak_value - threshold) / (threshold * margin), 0.0), 1.0))

            events.append({
                "anomaly_start_idx": int(anomaly_start_idx),
                "threshold_crossing_idx": int(threshold_crossing_idx),
                "anomaly_start_time": pd.Timestamp(timestamps[anomaly_start_idx]),
                "threshold_crossing_time": pd.Timestamp(timestamps[threshold_crossing_idx]),
                "peak_idx": int(peak_idx),
                "peak_time": pd.Timestamp(timestamps[peak_idx]),
                "end_idx": int(current_end),
                "end_time": pd.Timestamp(timestamps[current_end]),
                "confidence": confidence
            })

            anomalies[s] = events

        self.detected_anomalies = anomalies
        return dict(anomalies)

    def _find_anomaly_boundaries(self, error_series: np.ndarray, threshold: float, 
                                event_start_idx: int, event_end_idx: int) -> Tuple[int, int]:
        """
        Find the actual anomaly start and first threshold crossing within an event.
        - threshold_crossing_idx: first index in [event_start_idx, event_end_idx] with error > threshold
        - anomaly_start_idx: last index before crossing where smoothed error <= start_threshold,
        where start_threshold < threshold (to avoid collapsing to crossing).
        """

        # === 1) Threshold crossing ===
        threshold_crossing_idx = event_start_idx
        for idx in range(event_start_idx, event_end_idx + 1):
            if error_series[idx] > threshold:
                threshold_crossing_idx = idx
                break

        # === 2) Compute baseline in a short lookback window ===
        baseline_window = 8
        start_search = max(0, threshold_crossing_idx - baseline_window)
        anomaly_start_idx = threshold_crossing_idx  # default

        if start_search < threshold_crossing_idx:
            baseline_slice = error_series[start_search:threshold_crossing_idx]
            if baseline_slice.size == 0:
                return anomaly_start_idx, threshold_crossing_idx

            baseline = float(np.median(baseline_slice))
            # robust sigma (MAD) for stability
            mad = float(np.median(np.abs(baseline_slice - baseline))) or 0.0
            robust_sigma = 1.4826 * mad

            # --- Start threshold must be BELOW the anomaly threshold ---
            # Use the min of (0.8*threshold) and a robust baseline band
            start_threshold = min(0.8 * float(threshold), baseline + 3.0 * robust_sigma)

            # Tiny smoothing to reduce jitter (window=3, causal)
            def smooth3(arr, i):
                a = max(0, i - 2)
                b = i + 1
                return float(np.mean(arr[a:b]))

            found = False
            for idx in range(threshold_crossing_idx - 1, start_search - 1, -1):
                if smooth3(error_series, idx) <= start_threshold:
                    anomaly_start_idx = idx + 1  # first index above start_threshold
                    found = True
                    break
            if not found:
                anomaly_start_idx = start_search

        # ✅ Minimal safeguard: avoid start==crossing only if it happens
        if anomaly_start_idx >= threshold_crossing_idx:
            anomaly_start_idx = max(0, threshold_crossing_idx - 1)

        return anomaly_start_idx, threshold_crossing_idx


    def calculate_delays(self, sensor_data, use_case: str = "default"):
        if not self.detected_anomalies:
            self.detect_anomalies(sensor_data)

        if self.config["use_human_delay"] and self.config["human_review_figures"]:
            self.human_adjusted_delays = self.plot_human_review_figures(sensor_data, use_case)

        # ✅ Single-sensor case
        # if len(sensor_data) == 1:
        #     sensor_name = list(sensor_data.keys())[0]
        #     series = sensor_data[sensor_name]["value"].values
        #     if len(series) > 500_000:
        #         step = len(series) // 500_000
        #         series = series[::step]
        #     lag = self._estimate_delay(series, series)
        #     delay_seconds = int(lag)

        #     self.baseline_delays[sensor_name] = 0.0
        #     self.onsite_delays[sensor_name] = delay_seconds
        #     self.unadjusted_delays[sensor_name] = delay_seconds
        #     self.calibrated_delays[sensor_name] = delay_seconds

        #     print(f"\n📋 Onsite Delay Summary:")
        #     print(f"  {sensor_name}: Lab=0.00s, Onsite={delay_seconds:.2f}s, Final={delay_seconds:.2f}s")
        #     return {
        #         "lab_baseline_delays": self.baseline_delays,
        #         "measured_onsite_delays": self.onsite_delays,
        #         "lab+onsite_delays": self.unadjusted_delays,
        #         "final_calibrated_delays_with_reference_sensor": self.calibrated_delays,
        #         "human_adjusted_delays": self.human_adjusted_delays
        #     }

        # 🔄 Multi-sensor case (original logic restored)
        delays = {}
        delay_limit = self.config.get("delay_acceptance_limit", 60)
        lab_weight, onsite_weight = self.config.get("lab_onsite_weight", [0.5, 0.5])

        ref_sensor_id = self.config.get("reference_sensor_id")
        ref_delay = self.baseline_delays.get(ref_sensor_id) if ref_sensor_id else None
        ref_strategy = self.config.get("reference_strategy", "median_consensus")
        
        print("\n⏰ Onsite Delay Measurement Results:")

        for sensor_id, events in self.detected_anomalies.items():
            # Skip IRS sensors
            sensor_type = self.sensor_metadata.get(sensor_id) if self.sensor_metadata else None
            if sensor_type and sensor_type.upper() == "IRS":
                self.onsite_delays[sensor_id] = 0.0
                self.unadjusted_delays[sensor_id] = 0.0
                self.calibrated_delays[sensor_id] = 0.0
                print(f"  📊 {sensor_id}: 0.00s (IRS sensor - no delay)")
                continue

            if not events:
                print(f"  ⚠️ {sensor_id}: No events detected for onsite measurement")
                continue

            event_delays = []
            for event in events:

                start_time = event.get('anomaly_start_time') or event.get('start_time')
                # Prefer threshold crossing to match lab definition (first detection)
                end_time = event.get('threshold_crossing_time') or event.get('peak_time') or event.get('stabilization_time')
                if start_time is None or end_time is None:
                    continue

                delay = (end_time.replace(tzinfo=None) - start_time.replace(tzinfo=None)).total_seconds()

                # Use configured limit (defaults to 60s in self.config)
                if abs(delay) > delay_limit:
                    continue

                event_delays.append(delay)

            if event_delays:
                q1 = np.percentile(event_delays, 25)
                q3 = np.percentile(event_delays, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_delays = [d for d in event_delays if lower_bound <= d <= upper_bound]
                avg_delay = np.mean(filtered_delays) if filtered_delays else np.mean(event_delays)

                self.onsite_delays[sensor_id] = avg_delay
                print(f"  📊 {sensor_id}: {avg_delay:.2f}s (from {len(event_delays)} events)")

                baseline = self.baseline_delays.get(sensor_id, 0.0)
                final_delay = lab_weight * baseline + onsite_weight * avg_delay if baseline else avg_delay
                self.unadjusted_delays[sensor_id] = final_delay

                if self.config["use_reference_sensor"] and ref_delay and sensor_id != ref_sensor_id:
                    final_delay = self.apply_reference_strategy(final_delay, ref_delay, ref_strategy)

                # if sensor_id in self.human_adjusted_delays:
                #     human_delay = self.human_adjusted_delays[sensor_id]
                #     if abs(human_delay) >= 0.5:
                #         final_delay = 0.5 * final_delay + 0.5 * human_delay
                if sensor_id in self.human_adjusted_delays:
                    human_delay = self.human_adjusted_delays[sensor_id]
                    if abs(human_delay) >= 0.5:
                        old_final = final_delay
                        final_delay = 0.5 * final_delay + 0.5 * human_delay
                        print(f"  🧑 Human adjustment for {sensor_id}: {old_final:.2f}s → {final_delay:.2f}s (human={human_delay:.2f}s)")
                    else:
                        print(f"  🧑 Human delay for {sensor_id} = {human_delay:.2f}s (ignored, below threshold)")


                self.calibrated_delays[sensor_id] = final_delay
        
        # print(f"[DEBUG] Processing {sensor_id}: {len(events)} events, "
        #     f"baseline={self.baseline_delays.get(sensor_id, 'NA')}, "
        #     f"ref={ref_sensor_id if self.config['use_reference_sensor'] else 'None'}")
        
        # Preserve baseline for sensors without onsite events
        for sensor_id in self.baseline_delays:
            if sensor_id not in self.calibrated_delays:
                baseline_delay = self.baseline_delays[sensor_id]
                self.calibrated_delays[sensor_id] = baseline_delay
                print(f"  📊 {sensor_id}: Using baseline delay {baseline_delay:.2f}s (no onsite events)")

        # Print summary
        print(f"\n📋 Onsite Delay Summary:")
        for sensor_id in self.baseline_delays.keys():
            baseline = self.baseline_delays.get(sensor_id, 0.0)
            onsite = self.onsite_delays.get(sensor_id, 0.0)
            final = self.calibrated_delays.get(sensor_id, baseline)
            human = self.human_adjusted_delays.get(sensor_id, None)
            if human is not None:
                print(f"  {sensor_id}: Lab={baseline:.2f}s, Onsite={onsite:.2f}s, Human={human:.2f}s, Final={final:.2f}s")
            else:
                print(f"  {sensor_id}: Lab={baseline:.2f}s, Onsite={onsite:.2f}s, Final={final:.2f}s")

            # print(f"  {sensor_id}: Lab={baseline:.2f}s, Onsite={onsite:.2f}s, Final={final:.2f}s")
            # Build onsite+human blended delays
            onsite_human_delays = {}
            for sensor_id, onsite_delay in self.onsite_delays.items():
                human_delay = self.human_adjusted_delays.get(sensor_id)
                if human_delay is not None and abs(human_delay) >= 0.5:
                    blended = 0.5 * onsite_delay + 0.5 * human_delay
                else:
                    blended = onsite_delay
                onsite_human_delays[sensor_id] = blended
        self.onsite_human_delays = onsite_human_delays  
        return {
            "lab_baseline_delays": self.baseline_delays,
            "measured_onsite_delays": self.onsite_delays,
            "lab+onsite_delays": self.unadjusted_delays,
            "final_calibrated_delays_with_reference_sensor": self.calibrated_delays,
            "human_adjusted_delays": self.human_adjusted_delays,
            "onsite+human_delays": self.onsite_human_delays  
        }

    def _estimate_delay(self, signal_a, signal_b, max_lag=300):
        """Memory-safe cross-correlation delay estimator (for single-sensor)."""
        a = signal_a - np.mean(signal_a)
        b = signal_b - np.mean(signal_b)

        corr = correlate(a, b, mode="full", method="fft")
        lags = np.arange(-len(a) + 1, len(b))

        # restrict to ±max_lag samples
        mid = len(corr) // 2
        corr = corr[mid - max_lag: mid + max_lag + 1]
        lags = np.arange(-max_lag, max_lag + 1)

        return lags[np.argmax(corr)]

    def apply_reference_strategy(self, target_delay, ref_delay, strategy):
        if strategy == "median_consensus":
            return np.median([target_delay, ref_delay])
        elif strategy == "fast_response":
            return min(target_delay, ref_delay)
        else:
            return target_delay

    def _find_stabilization_point(self, data: pd.DataFrame, end_idx: int, window: int, threshold: float) -> Optional[int]:
        if end_idx + window >= len(data):
            return None

        for i in range(end_idx + 1, len(data) - window):
            window_variance = np.var(data['value'].iloc[i:i + window])
            if window_variance < threshold:
                return i

        return None

    def _calculate_dynamic_thresholds(self, sensor_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        thresholds = {}
        for sensor_id, data in sensor_data.items():
            std_dev = np.std(data['value'])
            thresholds[sensor_id] = max(self.config["minimum_anomaly_magnitude"], 3 * std_dev)
        return thresholds

    def plot_human_review_figures(self, sensor_data: Dict[str, pd.DataFrame], use_case: str = "default") -> Dict[str, float]:
        output_dir = os.path.join("figures", use_case, "human_review")
        os.makedirs(output_dir, exist_ok=True)
        human_adjusted_delays = {}
        review_log = []


        window = self.config.get("human_review_window_sec", 60)
        max_samples = self.config.get("num_human_review_samples")  # Can be None
        only_uncertain = self.config.get("only_uncertain_events", False)

        for sensor_id, events in self.detected_anomalies.items():
            if only_uncertain:
                events = [e for e in events if e.get("confidence", 1.0) < self.config.get("confidence_threshold", 0.8)]
            if not events:
                print(f"⚠️ No events selected for review for sensor {sensor_id}")
                continue

            sample_limit = len(events) if max_samples is None else min(len(events), max_samples)
            reviewed = 0

            for event_id, event in enumerate(events[:sample_limit], start=1):
                print(f"🧪 Event for {sensor_id} has confidence {event.get('confidence', 1.0):.2f}")

                ts_series = sensor_data[sensor_id]['timestamp']
                val_series = sensor_data[sensor_id]['value']
                # center_time = event['start_time']
                center_time = event.get('anomaly_start_time') or event.get('start_time')
                window_half = pd.Timedelta(seconds=window / 2)
                start_window = max(center_time - window_half, ts_series.iloc[0])
                end_window = min(center_time + window_half, ts_series.iloc[-1])
                window_mask = (ts_series >= start_window) & (ts_series <= end_window)

                actual_time = None

                def onclick(event_click):
                    nonlocal actual_time
                    if event_click.xdata:
                        actual_time = mdates.num2date(event_click.xdata)
                        plt.close()

                # Initial plot for click
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(ts_series[window_mask], val_series[window_mask], label=f'{sensor_id} values')
                # ax.axvline(center_time, color='red', linestyle='--', label='Detected Start Time')
                ax.axvline(center_time, color='red', linestyle='--', label='Algorithm Detection Time')
                ax.set_xlim([start_window, end_window])
                # ax.set_title(f"Click to select actual start time for {sensor_id} event {event_id}")
                ax.set_title(f"Click when the algorithm SHOULD have detected {sensor_id} event {event_id}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Sensor Value")
                ax.legend()
                ax.grid(True)
                fig.autofmt_xdate()

                cid = fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()

                if actual_time is None:
                    print("⚠️ No click detected. Using detected start_time.")
                    actual_time = center_time

                try:
                    detected_start = center_time.replace(tzinfo=None)
                    selected_time = actual_time.replace(tzinfo=None)
                    # delay_sec = (detected_start - selected_time).total_seconds()
                    delay_sec = (selected_time - detected_start).total_seconds()

                    print(f"📍 Detected start: {detected_start}, Your selected time: {selected_time}")
                    print(f"✅ Delay (model - actual): {delay_sec:.2f} seconds")

                    # Save annotated figure with both lines
                    fig2, ax2 = plt.subplots(figsize=(12, 5))
                    ax2.plot(ts_series[window_mask], val_series[window_mask], label=f'{sensor_id} values')
                    ax2.axvline(center_time, color='red', linestyle='--', label='Detected Start Time')
                    ax2.axvline(actual_time, color='blue', linestyle=':', label='Human Click Time')
                    ax2.set_xlim([start_window, end_window])
                    conf = event.get('confidence', 1.0)
                    color = 'green' if conf >= 0.8 else 'orange' if conf >= 0.5 else 'red'
                    ax2.set_title(f"{sensor_id} Event {event_id} | Confidence: {conf:.2f}", color=color)
                    ax2.set_xlabel("Time")
                    ax2.set_ylabel("Sensor Value")
                    ax2.legend()
                    ax2.grid(True)
                    fig2.autofmt_xdate()
                    # fname = f"figures/human_review/{sensor_id}_event{event_id}.png"
                    fname = os.path.join(output_dir, f"{sensor_id}_event{event_id}.png")
                    fig2.savefig(fname)
                    plt.close(fig2)
                    print(f"📷 Saved figure to {fname}")

                    human_adjusted_delays.setdefault(sensor_id, []).append(delay_sec)
                    reviewed += 1

                    # Save for CSV
                    review_log.append({
                        "sensor_id": sensor_id,
                        "event_id": event_id,
                        "detected_start_time": detected_start,
                        "human_click_time": selected_time,
                        "delay_sec": delay_sec,
                        "confidence": round(conf, 2)
                    })

                except Exception as e:
                    print(f"❌ Error calculating delay: {e}")
                    continue

        # Save review log to CSV
        if review_log:
            df_log = pd.DataFrame(review_log)
            # df_log.to_csv("figures/human_reviewed_delays.csv", index=False)
            df_log.to_csv(os.path.join(output_dir, "reviewed_delays.csv"), index=False)
            print("📝 Saved delay log to reviewed_delays.csv")

        return {s: np.mean(dlist) for s, dlist in human_adjusted_delays.items()}

    def save_delays(self, filepath: str = 'calibrated_delays.json') -> None:
        all_delays = {
            "lab_baseline_delays": self.baseline_delays,
            "measured_onsite_delays": self.onsite_delays,
            "lab+onsite_delays": self.unadjusted_delays,
            "final_calibrated_delays_with_reference_sensor": self.calibrated_delays,
            "human_adjusted_delays": self.human_adjusted_delays,
            "onsite+human_delays": self.onsite_human_delays
        }


        try:
            with open(filepath, 'w') as f:
                json.dump(all_delays, f, indent=2)
            logger.info(f"Delays saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving delays: {e}")
        
    def load_delays(self, filepath: str = 'calibrated_delays.json') -> None:
        try:
            with open(filepath, 'r') as f:
                all_delays = json.load(f)

            if "baseline_delays" in all_delays:
                self.baseline_delays = all_delays["baseline_delays"]
            if "onsite_delays" in all_delays:
                self.onsite_delays = all_delays["onsite_delays"]
            if "unadjusted_delays" in all_delays:
                self.unadjusted_delays = all_delays["unadjusted_delays"]
            if "final_calibrated_delays" in all_delays:
                self.calibrated_delays = all_delays["final_calibrated_delays"]
            if "human_adjusted_delays" in all_delays:
                self.human_adjusted_delays = all_delays["human_adjusted_delays"]

            logger.info(f"Delays loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Delay file {filepath} not found")
        except Exception as e:
            logger.error(f"Error loading delays: {e}")

    def export_all_delay_metrics(self):
        all_delays = {
            "lab_baseline_delays": self.baseline_delays,
            "measured_onsite_delays": self.onsite_delays,
            "Lab+onsite_delays": self.unadjusted_delays,
            "final_calibrated_delays": self.calibrated_delays,
            "human_adjusted_delays": self.human_adjusted_delays,
            "onsite+human_delays": self.onsite_human_delays        
            }
        return all_delays
