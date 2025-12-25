"""
Real-Time Alert Generator Module

Generates alerts for detected anomalies with temporal segmentation.
Prevents alert spam by grouping consecutive anomalies.
Supports multiple alert integrations (console, MQTT, HTTP webhooks).
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class RealtimeAlertManager:
    """
    Real-time alert generation and management.
    
    Features:
    - Temporal segmentation (anomalies with gap > 5 min = new alert)
    - Alert deduplication
    - Multi-channel delivery (console, optional MQTT/HTTP)
    - Per-use-case messages and locations
    """
    
    # Message templates per use case
    ANOMALY_MESSAGES = {
        "door": "door left open",
        "co2": "unauthorized person detected",
        "appliance": "unattended appliance",
        "abnormal_object": "object blocking hallway"
    }
    
    # Location templates per use case
    LOCATIONS = {
        "door": "Entrance",
        "co2": "House",
        "appliance": "Kitchen",
        "abnormal_object": "Hallway"
    }
    
    def __init__(self, config):
        """
        Initialize alert manager.
        
        Args:
            config: LiveInferenceConfig instance
        """
        self.config = config
        self.use_case = config.use_case
        self.segmentation_gap = timedelta(minutes=5)  # 5-minute gap between segments
        
        self.active_segment: Optional[Dict] = None
        self.alerts_generated = 0
        self.alert_history = []  # Keep track of recent alerts
        
        logger.info(f"🚨 Alert manager initialized for {self.use_case}")
    
    def get_message(self) -> str:
        """Get anomaly message for use case."""
        return self.ANOMALY_MESSAGES.get(self.use_case, "anomaly detected")
    
    def get_location(self) -> str:
        """Get location for use case."""
        return self.LOCATIONS.get(self.use_case, "Room")
    
    def process_inference_result(
        self,
        sensor_inference: Dict,
        vision_inference: Optional[Dict] = None,
        timestamp: Optional[str] = None
    ) -> Optional[str]:
        """
        Process a single inference result and generate alert if needed.
        
        Uses temporal segmentation: consecutive anomalies are grouped into
        segments separated by gaps > 5 minutes.
        
        Args:
            sensor_inference: Result from sensor inference
            vision_inference: Optional result from vision fallback
            timestamp: ISO timestamp string
            
        Returns:
            Alert message string, or None if no alert
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        try:
            timestamp_dt = datetime.fromisoformat(timestamp)
        except:
            timestamp_dt = datetime.now()
        
        # Determine if this is an anomaly
        is_anomaly = self._is_anomaly(sensor_inference, vision_inference)
        confidence = self._get_confidence(sensor_inference, vision_inference)
        
        # Update current segment
        if is_anomaly:
            if self.active_segment is None:
                # Start new segment
                self.active_segment = {
                    "start_time": timestamp_dt,
                    "last_time": timestamp_dt,
                    "max_confidence": confidence,
                    "count": 1,
                    "anomalies": [sensor_inference]
                }
                logger.debug(f"🔴 Starting anomaly segment at {timestamp}")
            else:
                # Check if this is part of existing segment (gap <= 5 min)
                time_since_last = timestamp_dt - self.active_segment["last_time"]
                
                if time_since_last <= self.segmentation_gap:
                    # Add to existing segment
                    self.active_segment["last_time"] = timestamp_dt
                    self.active_segment["max_confidence"] = max(
                        self.active_segment["max_confidence"],
                        confidence
                    )
                    self.active_segment["count"] += 1
                    self.active_segment["anomalies"].append(sensor_inference)
                    logger.debug(f"🔴 Continuing segment (count={self.active_segment['count']})")
                else:
                    # Gap too large, start new segment and generate alert for old
                    alert = self._generate_segment_alert(self.active_segment)
                    
                    # Reset for new segment
                    self.active_segment = {
                        "start_time": timestamp_dt,
                        "last_time": timestamp_dt,
                        "max_confidence": confidence,
                        "count": 1,
                        "anomalies": [sensor_inference]
                    }
                    
                    return alert
        
        else:
            # Normal reading
            if self.active_segment is not None:
                # End of segment, generate alert
                time_since_last = timestamp_dt - self.active_segment["last_time"]
                
                if time_since_last > self.segmentation_gap:
                    alert = self._generate_segment_alert(self.active_segment)
                    self.active_segment = None
                    return alert
            
            logger.debug(f"✅ Normal reading at {timestamp}")
        
        return None
    
    def _is_anomaly(
        self,
        sensor_inference: Dict,
        vision_inference: Optional[Dict]
    ) -> bool:
        """Determine if reading is anomalous."""
        # Sensor anomaly
        if sensor_inference.get("status") == "Anomaly":
            return True
        
        # Vision anomaly (if fallback ran)
        if vision_inference and vision_inference.get("vision_status") == "Anomaly":
            return True
        
        return False
    
    def _get_confidence(
        self,
        sensor_inference: Dict,
        vision_inference: Optional[Dict]
    ) -> float:
        """Get maximum confidence from sensor or vision."""
        confidence = sensor_inference.get("confidence", 0.0)
        
        if vision_inference:
            vision_conf = vision_inference.get("vision_confidence", 0.0)
            confidence = max(confidence, vision_conf)
        
        return confidence
    
    def _generate_segment_alert(self, segment: Dict) -> str:
        """
        Generate alert message for a completed anomaly segment.
        
        Args:
            segment: Dict with segment info (start_time, max_confidence, count)
            
        Returns:
            Alert message string
        """
        start_time = segment["start_time"]
        max_confidence = segment["max_confidence"]
        duration = segment["last_time"] - segment["start_time"]
        
        time_str = start_time.strftime("%H:%M")
        location = self.get_location()
        message = self.get_message()
        confidence_pct = int(max_confidence * 100)
        
        # Format alert message
        alert_msg = (
            f"🚨 ALERT: **{location}: {message}, "
            f"{confidence_pct}% confidence – {time_str}** "
            f"(duration: {int(duration.total_seconds())}s, events: {segment['count']})"
        )
        
        self.alerts_generated += 1
        self.alert_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": alert_msg,
            "location": location,
            "confidence": max_confidence,
            "duration_seconds": int(duration.total_seconds()),
            "event_count": segment['count']
        })
        
        logger.warning(alert_msg)
        
        return alert_msg
    
    def flush_active_segment(self) -> Optional[str]:
        """
        Flush any active segment and generate alert.
        
        Call this at shutdown or after a timeout to ensure last segment is alerted.
        
        Returns:
            Alert message, or None if no active segment
        """
        if self.active_segment is not None:
            alert = self._generate_segment_alert(self.active_segment)
            self.active_segment = None
            return alert
        
        return None
    
    def send_alert(
        self,
        alert_message: str,
        channels: List[str] = None
    ) -> bool:
        """
        Send alert through configured channels.
        
        Args:
            alert_message: Alert text
            channels: List of channels ['console', 'mqtt', 'http', 'sms']
            
        Returns:
            True if at least one channel succeeded
        """
        if channels is None:
            channels = ['console']
        
        success = False
        
        for channel in channels:
            try:
                if channel == 'console':
                    self._send_console_alert(alert_message)
                    success = True
                
                elif channel == 'mqtt':
                    if self._send_mqtt_alert(alert_message):
                        success = True
                
                elif channel == 'http':
                    if self._send_http_alert(alert_message):
                        success = True
                
                elif channel == 'sms':
                    if self._send_sms_alert(alert_message):
                        success = True
            
            except Exception as e:
                logger.error(f"❌ Failed to send alert via {channel}: {e}")
        
        return success
    
    def _send_console_alert(self, message: str):
        """Send alert to console (already logged above)."""
        pass
    
    def _send_mqtt_alert(self, message: str) -> bool:
        """
        Send alert via MQTT (stub for future integration).
        
        TODO: Implement MQTT broker connection
        """
        try:
            # Placeholder for MQTT implementation
            # import paho.mqtt.client as mqtt
            # client = mqtt.Client()
            # client.connect(self.config.mqtt_broker_host, self.config.mqtt_broker_port)
            # client.publish(f"alerts/{self.use_case}", message)
            # client.disconnect()
            logger.debug("📨 [MQTT] Alert would be sent (not implemented)")
            return False
        except Exception as e:
            logger.error(f"❌ MQTT alert failed: {e}")
            return False
    
    def _send_http_alert(self, message: str) -> bool:
        """
        Send alert via HTTP webhook (stub for future integration).
        
        TODO: Implement HTTP webhook
        """
        try:
            # Placeholder for HTTP implementation
            # import requests
            # payload = {"alert": message, "use_case": self.use_case}
            # requests.post(self.config.http_webhook_url, json=payload)
            logger.debug("📨 [HTTP] Alert would be sent (not implemented)")
            return False
        except Exception as e:
            logger.error(f"❌ HTTP alert failed: {e}")
            return False
    
    def _send_sms_alert(self, message: str) -> bool:
        """
        Send alert via SMS (stub for future integration).
        
        TODO: Implement SMS gateway
        """
        try:
            # Placeholder for SMS implementation
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(body=message, from_=from_num, to=to_num)
            logger.debug("📨 [SMS] Alert would be sent (not implemented)")
            return False
        except Exception as e:
            logger.error(f"❌ SMS alert failed: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Return alert statistics."""
        return {
            "alerts_generated": self.alerts_generated,
            "alert_history_count": len(self.alert_history),
            "active_segment": self.active_segment is not None,
            "active_segment_duration_s": (
                (self.active_segment["last_time"] - self.active_segment["start_time"]).total_seconds()
                if self.active_segment else 0
            ),
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return self.alert_history[-limit:]
