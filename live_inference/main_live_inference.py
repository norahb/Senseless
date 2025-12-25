"""
Main Live Inference Orchestrator

Ties all components together:
  - Arduino serial reading
  - Sensor model inference
  - Confidence estimation
  - Vision fallback
  - Alert generation
  - Incoming sensor data logging for retraining

Single-sample streaming pipeline with model loading at startup.
"""

import os
import sys
import csv
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict
import signal

# Add training path for imports
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
if training_path not in sys.path:
    sys.path.insert(0, training_path)

from live_inference.config.config_live import LiveInferenceConfig
from live_inference.arduino_serial_reader import ArduinoSerialReader
from live_inference.sensor_inference_realtime import RealtimeSensorInference
from live_inference.vision_fallback_realtime import RealtimeVisionFallback
from live_inference.alerts_realtime import RealtimeAlertManager

# Setup logging
def configure_logging(level_name: str = "INFO"):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

configure_logging()
logger = logging.getLogger(__name__)


class LiveInferencePipeline:
    """
    Main live inference pipeline orchestrator.
    
    Orchestrates:
    1. Arduino serial reading (with auto-reconnect)
    2. Single-sample sensor inference (model loaded at startup)
    3. Confidence estimation
    4. Vision fallback (on-demand image capture)
    5. Alert generation (with temporal segmentation)
    6. Incoming sensor data logging (for retraining)
    """
    
    def __init__(
        self,
        use_case: str,
        arduino_port: Optional[str] = None,
        enable_vision_fallback: bool = True
    ):
        """
        Initialize live inference pipeline.
        
        Args:
            use_case: Use case identifier (door, co2, appliance, abnormal_object)
            arduino_port: Serial port for Arduino (auto-detected if None)
            enable_vision_fallback: Whether to enable vision fallback
        """
        self.use_case = use_case
        self.enable_vision_fallback = enable_vision_fallback
        self.is_running = False
        
        # Initialize configuration
        logger.info(f"📋 Initializing live inference for {use_case}...")
        self.config = LiveInferenceConfig(use_case, arduino_port=arduino_port)
        
        # Log configuration
        config_summary = self.config.get_live_config_summary()
        # logger.info("Live Inference Config:")
        # logger.info(config_summary)

        # Log resolved model paths for quick diagnostics
        sensor_model_dir = (config_summary.get("models", {}) or {}).get("sensor_model_dir")
        vision_model_path = (config_summary.get("models", {}) or {}).get("vision_model_path")
        logger.info(f"Resolved sensor model dir: {sensor_model_dir}")
        logger.info(f"Resolved vision model path: {vision_model_path}")
        
        # Initialize Arduino reader
        logger.info(f"🔌 Initializing Arduino reader on {self.config.arduino_port}...")
        self.arduino_reader = ArduinoSerialReader(
            port=self.config.arduino_port,
            baudrate=self.config.arduino_baudrate,
            timeout=self.config.arduino_timeout,
            sensor_cols=self.config.use_case_config.sensor_cols,
            max_retry_delay=self.config.arduino_max_retry_delay,
            heartbeat_interval=self.config.arduino_heartbeat_interval
        )
        
        # Initialize sensor inference (loads model at startup)
        logger.info(f"🧠 Initializing sensor inference (model loaded at startup)...")
        self.sensor_inference = RealtimeSensorInference(self.config)
        
        # Initialize vision fallback (loads model at startup)
        if self.enable_vision_fallback:
            logger.info(f"📸 Initializing vision fallback (model loaded at startup)...")
            self.vision_fallback = RealtimeVisionFallback(self.config, camera_source=self.config.camera_source)
        else:
            self.vision_fallback = None
            logger.info(f"⏭️  Vision fallback disabled")
        
        # Initialize alert manager
        logger.info(f"🚨 Initializing alert manager...")
        self.alert_manager = RealtimeAlertManager(self.config)
        
        # Setup sensor data logging (for retraining)
        self._setup_sensor_data_logging()
        
        # Statistics
        self.samples_processed = 0
        self.anomalies_detected = 0
        self.vision_fallbacks_triggered = 0
        self.alerts_generated = 0
        
        # Sensor history for outlier detection (rolling window)
        self.sensor_history = {col: [] for col in self.config.use_case_config.sensor_cols}
        self.max_history_size = 100
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"✅ Live inference pipeline initialized for {use_case}")
    
    def _setup_sensor_data_logging(self):
        """Setup CSV writer for incoming sensor data (for retraining)."""
        self.sensor_data_log_path = self.config.sensor_data_log_path
        
        # Create directory
        os.makedirs(os.path.dirname(self.sensor_data_log_path), exist_ok=True)
        
        # Check if file exists (for appending)
        file_exists = os.path.exists(self.sensor_data_log_path)
        
        # Open file for appending
        self.sensor_data_file = open(self.sensor_data_log_path, 'a', newline='')
        
        # Setup CSV writer with headers
        sensor_cols = self.config.use_case_config.sensor_cols
        fieldnames = sensor_cols + [
            'Timestamp',
            'Sensor_Status',
            'Sensor_Confidence',
            'Vision_Status',
            'Vision_Confidence',
            'Alert_Generated'
        ]
        
        self.sensor_data_writer = csv.DictWriter(
            self.sensor_data_file,
            fieldnames=fieldnames
        )
        
        # Write header if file is new
        if not file_exists:
            self.sensor_data_writer.writeheader()
            logger.info(f"📝 Created sensor data log: {self.sensor_data_log_path}")
        else:
            logger.info(f"📝 Appending to sensor data log: {self.sensor_data_log_path}")
    
    def _update_sensor_history(self, sensor_data: Dict[str, float]):
        """Update rolling window history for outlier detection."""
        for col in self.config.use_case_config.sensor_cols:
            if col in sensor_data:
                self.sensor_history[col].append(sensor_data[col])
                
                # Keep only last N samples
                if len(self.sensor_history[col]) > self.max_history_size:
                    self.sensor_history[col].pop(0)
    
    def _log_sensor_data(
        self,
        sensor_data: Dict[str, float],
        timestamp: str,
        sensor_result: Dict,
        vision_result: Optional[Dict],
        alert_generated: bool
    ):
        """Log incoming sensor data to CSV for retraining."""
        try:
            sensor_cols = self.config.use_case_config.sensor_cols
            
            row = {col: sensor_data.get(col, None) for col in sensor_cols}
            row.update({
                'Timestamp': timestamp,
                'Sensor_Status': sensor_result.get('status', 'Unknown'),
                'Sensor_Confidence': sensor_result.get('confidence', 0.0),
                'Vision_Status': vision_result.get('vision_status', 'N/A') if vision_result else 'N/A',
                'Vision_Confidence': vision_result.get('vision_confidence', 0.0) if vision_result else 0.0,
                'Alert_Generated': 1 if alert_generated else 0
            })
            
            self.sensor_data_writer.writerow(row)
            self.sensor_data_file.flush()
            
            logger.debug(f"📝 Logged sensor data: {row}")
        
        except Exception as e:
            logger.error(f"❌ Failed to log sensor data: {e}")
    
    def process_single_sample(self) -> bool:
        """
        Process a single sensor sample through the entire pipeline.
        
        Returns:
            True if sample was processed, False if read failed
        """
        # Step 1: Read from Arduino
        read_result = self.arduino_reader.read_sample()
        if read_result is None:
            return False
        
        sensor_data, timestamp = read_result
        self.samples_processed += 1
        
        if not self.config.compact_output:
            logger.info(f"\n📊 Sample #{self.samples_processed}: {sensor_data} @ {timestamp}")
        
        # Step 2: Update sensor history (for outlier detection)
        self._update_sensor_history(sensor_data)
        
        # Step 3: Run sensor inference
        sensor_result = self.sensor_inference.infer_sample(
            sensor_data,
            sensor_history=self.sensor_history
        )
        if not self.config.compact_output:
            logger.info(f"   Sensor: {sensor_result['status']} (conf={sensor_result['confidence']:.3f})")
        
        # Track anomalies
        if sensor_result['status'] == 'Anomaly':
            self.anomalies_detected += 1
        
        # Step 4: Run vision fallback (if configured and triggered)
        vision_result = None
        triggered = False
        if (
            self.enable_vision_fallback and
            (sensor_result['confidence'] < self.config.confidence_threshold_for_vision_fallback or
             sensor_result['status'] == 'Sensor_Error')
        ):
            self.vision_fallbacks_triggered += 1
            triggered = True
            if not self.config.compact_output:
                logger.info(f"   Triggering vision fallback...")
            vision_result = self.vision_fallback.run_fallback(sensor_result)
            if not self.config.compact_output:
                logger.info(f"   Vision: {vision_result['vision_status']} (conf={vision_result['vision_confidence']:.3f})")
        
        # Step 5: Generate alerts (with temporal segmentation)
        alert_msg = self.alert_manager.process_inference_result(
            sensor_result,
            vision_result,
            timestamp
        )
        
        alert_generated = alert_msg is not None
        if alert_generated:
            self.alerts_generated += 1
            # Send alert through configured channels
            self.alert_manager.send_alert(alert_msg, channels=['console'])
        
        # Step 6: Log sensor data (for retraining)
        if self.config.log_incoming_sensor_data:
            self._log_sensor_data(
                sensor_data,
                timestamp,
                sensor_result,
                vision_result,
                alert_generated
            )
        
        # Compact summary output
        if self.config.compact_output:
            # 1- sensor readings
            sensor_str = ", ".join([f"{k}={sensor_data.get(k)}" for k in self.config.use_case_config.sensor_cols])
            print(f"Sensors: {sensor_str}")
            # 2- sensor prediction; 3- confidence
            print(f"Sensor: {sensor_result['status']} | Confidence: {sensor_result['confidence']:.3f}")
            # 4- flag of vision activation
            print(f"Vision activated: {triggered}")
            # 5- show camera opening and capturing image
            if triggered:
                cam_idx = getattr(self.vision_fallback, 'last_opened_index', self.config.camera_source)
                if vision_result and vision_result.get('vision_status') == 'Capture_Failed':
                    print(f"Camera: opened index {cam_idx}; capture failed")
                else:
                    print(f"Camera: opened index {cam_idx}; captured image")
            # 6- print image prediction
            if vision_result:
                print(f"Image: {vision_result['vision_status']} | Confidence: {vision_result['vision_confidence']:.3f}")
            # 7- show alert if needed
            if alert_generated and alert_msg:
                print(f"Alert: {alert_msg['type']} | {alert_msg.get('message','')} ")

        return True
    
    def run(self, max_samples: Optional[int] = None):
        """
        Run the live inference pipeline.
        
        Args:
            max_samples: Maximum number of samples to process (None for infinite)
        """
        self.is_running = True
        logger.info(f"🚀 Starting live inference pipeline...")
        logger.info(f"   Use case: {self.use_case}")
        logger.info(f"   Arduino port: {self.config.arduino_port}")
        logger.info(f"   Vision fallback: {self.enable_vision_fallback}")
        logger.info(f"   Max samples: {max_samples if max_samples else 'unlimited'}")
        
        start_time = time.time()
        
        try:
            while self.is_running:
                # Process sample
                if not self.process_single_sample():
                    # Read failed, wait and retry
                    time.sleep(0.1)
                    continue
                
                # Check if reached max samples
                if max_samples and self.samples_processed >= max_samples:
                    logger.info(f"✅ Reached max samples ({max_samples})")
                    break
        
        except KeyboardInterrupt:
            logger.info(f"⏹️  Interrupted by user")
        
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        
        finally:
            self.shutdown()
            
            # Print final statistics
            elapsed = time.time() - start_time
            self._print_statistics(elapsed)
    
    def shutdown(self):
        """Gracefully shutdown the pipeline."""
        logger.info(f"🛑 Shutting down live inference pipeline...")
        
        self.is_running = False
        
        # Flush any active alert segment
        remaining_alert = self.alert_manager.flush_active_segment()
        if remaining_alert:
            logger.warning(f"⚠️  Final alert: {remaining_alert}")
        
        # Close Arduino connection
        if self.arduino_reader:
            self.arduino_reader.close()
        
        # Close sensor data log file
        if self.sensor_data_file:
            self.sensor_data_file.close()
        
        logger.info(f"✅ Pipeline shutdown complete")
    
    def _signal_handler(self, sig, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"\n🛑 Received signal {sig}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def _print_statistics(self, elapsed_seconds: float):
        """Print comprehensive statistics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 LIVE INFERENCE STATISTICS")
        logger.info(f"{'='*60}")
        
        logger.info(f"\n⏱️  TIMING:")
        logger.info(f"   Total runtime: {elapsed_seconds:.1f}s")
        logger.info(f"   Samples processed: {self.samples_processed}")
        if self.samples_processed > 0:
            logger.info(f"   Avg sample rate: {self.samples_processed/elapsed_seconds:.2f} samples/sec")
        
        logger.info(f"\n🎯 SENSOR INFERENCE:")
        sensor_stats = self.sensor_inference.get_stats()
        logger.info(f"   Total inferences: {sensor_stats['inference_count']}")
        logger.info(f"   Anomalies detected: {self.anomalies_detected}")
        logger.info(f"   Sensor errors: {sensor_stats['total_sensor_errors']}")
        logger.info(f"   Outliers detected: {sensor_stats['total_outliers_detected']}")
        logger.info(f"   Error rate: {sensor_stats['error_rate']*100:.2f}%")
        
        if self.enable_vision_fallback:
            logger.info(f"\n📸 VISION FALLBACK:")
            vision_stats = self.vision_fallback.get_stats()
            logger.info(f"   Fallbacks triggered: {self.vision_fallbacks_triggered}")
            logger.info(f"   Vision inferences: {vision_stats['inference_count']}")
            logger.info(f"   Successful captures: {vision_stats['successful_captures']}")
            logger.info(f"   Failed captures: {vision_stats['failed_captures']}")
            logger.info(f"   Capture success rate: {vision_stats['capture_success_rate']*100:.1f}%")
            logger.info(f"   Avg inference time: {vision_stats['avg_inference_time_ms']:.1f}ms")
        
        logger.info(f"\n🚨 ALERTS:")
        alert_stats = self.alert_manager.get_stats()
        logger.info(f"   Alerts generated: {alert_stats['alerts_generated']}")
        logger.info(f"   Active segment: {alert_stats['active_segment']}")
        
        logger.info(f"\n🔌 ARDUINO:")
        arduino_stats = self.arduino_reader.get_stats()
        logger.info(f"   Connected: {arduino_stats['is_connected']}")
        logger.info(f"   Total reads: {arduino_stats['total_reads']}")
        logger.info(f"   Total errors: {arduino_stats['total_errors']}")
        logger.info(f"   Error rate: {arduino_stats['error_rate']*100:.2f}%")
        logger.info(f"   Seconds since last read: {arduino_stats['seconds_since_last_read']:.1f}s")
        
        logger.info(f"\n📝 LOGGING:")
        logger.info(f"   Sensor data log: {self.config.sensor_data_log_path}")
        logger.info(f"   Fallback images saved: {os.path.exists(self.config.live_fallback_images_dir)}")
        
        logger.info(f"\n{'='*60}")


def main(
    use_case: str = "door",
    arduino_port: Optional[str] = None,
    max_samples: Optional[int] = None,
    enable_vision_fallback: bool = True
):
    """
    Main entry point for live inference.
    
    Args:
        use_case: Use case identifier (door, co2, appliance, abnormal_object)
        arduino_port: Serial port for Arduino (auto-detected if None)
        max_samples: Max samples to process (None for infinite)
        enable_vision_fallback: Whether to enable vision fallback
    
    Example:
        python -m live_inference.main_live_inference --use_case door --max_samples 1000
    """
    pipeline = LiveInferencePipeline(
        use_case=use_case,
        arduino_port=arduino_port,
        enable_vision_fallback=enable_vision_fallback
    )
    
    pipeline.run(max_samples=max_samples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Live inference pipeline with Arduino sensor reading and vision fallback"
    )
    parser.add_argument(
        "--use_case",
        type=str,
        default="door",
        help="Use case identifier (supports aliases: doors→door)"
    )
    parser.add_argument(
        "--arduino_port",
        type=str,
        default=None,
        help="Arduino serial port (e.g., /dev/rfcomm0 or COM3). Auto-detected if not provided."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for infinite)"
    )
    parser.add_argument(
        "--no_vision_fallback",
        action="store_true",
        help="Disable vision fallback"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Enable compact per-sample output (minimal prints)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a brief camera preview window during capture"
    )
    parser.add_argument(
        "--preview_ms",
        type=int,
        default=None,
        help="Preview window duration in milliseconds (default 500)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Enable saving fallback images"
    )
    parser.add_argument(
        "--save_anomalies_only",
        action="store_true",
        help="Save only anomaly images when saving is enabled"
    )
    
    args = parser.parse_args()
    
    # Normalize use_case aliases
    use_case_arg = (args.use_case or "door").lower().strip()
    if use_case_arg == "doors":
        use_case_arg = "door"
    if use_case_arg not in ["door", "co2", "appliance", "abnormal_object"]:
        logger.warning(f"Unknown use_case '{args.use_case}', defaulting to 'door'")
        use_case_arg = "door"

    # Apply logging level
    configure_logging(args.log_level)

    # Initialize pipeline
    pipeline = LiveInferencePipeline(
        use_case=use_case_arg,
        arduino_port=args.arduino_port,
        enable_vision_fallback=not args.no_vision_fallback
    )

    # Enable compact output mode
    if args.compact:
        pipeline.config.compact_output = True
        # Reduce internal module logger noise
        logging.getLogger('live_inference.sensor_inference_realtime').setLevel(logging.ERROR)
        logging.getLogger('live_inference.vision_fallback_realtime').setLevel(logging.ERROR)
        logging.getLogger('live_inference.arduino_serial_reader').setLevel(logging.ERROR)
        logging.getLogger('live_inference.alerts_realtime').setLevel(logging.ERROR)

    # Enable preview window if requested
    if args.preview:
        pipeline.config.preview_capture = True
    if args.preview_ms is not None:
        pipeline.config.preview_delay_ms = args.preview_ms

    # Optional: override image persistence via CLI
    if args.save_images:
        pipeline.config.override_image_persistence(
            save=True,
            anomalies_only=args.save_anomalies_only
        )
        logger.info(f"💾 Image saving enabled (anomalies_only={args.save_anomalies_only})")

    # Run
    pipeline.run(max_samples=args.max_samples)
