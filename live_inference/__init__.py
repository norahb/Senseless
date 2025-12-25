"""
Live Inference Module

Real-time inference pipeline for streaming sensor data from Arduino.
Orchestrates:
  - Arduino serial data reading with auto-reconnect
  - Single-sample sensor model inference
  - Confidence estimation
  - Vision fallback on low confidence/sensor errors
  - Real-time alert generation
  - Sensor data logging for retraining
"""

__version__ = "1.0.0"
