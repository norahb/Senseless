"""
Arduino Serial Reader Module

Handles Bluetooth/USB serial communication with Arduino, auto-reconnect logic,
CSV sensor data parsing, and heartbeat checks.
"""

import serial
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ArduinoSerialReader:
    """
    Manages serial communication with Arduino sensors.
    
    Features:
    - Auto-reconnect with exponential backoff (1s → 30s)
    - CSV sensor data parsing
    - Heartbeat/keepalive detection
    - Timestamped sensor readings
    - Graceful handling of corrupted/incomplete lines
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: int = 5,
        sensor_cols: List[str] = None,
        max_retry_delay: int = 30,
        heartbeat_interval: int = 5
    ):
        """
        Initialize Arduino serial reader.
        
        Args:
            port: Serial port (e.g., '/dev/rfcomm0' on Linux, 'COM3' on Windows)
            baudrate: Baud rate (default 9600 for HC-05 Bluetooth modules)
            timeout: Serial read timeout in seconds
            sensor_cols: Expected sensor column names
            max_retry_delay: Max delay between reconnection attempts (seconds)
            heartbeat_interval: Expected data arrival interval (seconds)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.sensor_cols = sensor_cols or []
        self.max_retry_delay = max_retry_delay
        self.heartbeat_interval = heartbeat_interval
        
        self.serial_conn: Optional[serial.Serial] = None
        self.retry_delay = 1  # Start with 1 second
        self.last_successful_read_time = time.time()
        self.is_connected = False
        self.total_reads = 0
        self.total_errors = 0
        
        # Try to connect immediately
        self._connect()
    
    def _connect(self) -> bool:
        """
        Attempt to connect to Arduino via serial port.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.serial_conn is not None and self.serial_conn.is_open:
                self.serial_conn.close()
            
            logger.info(f"🔌 Connecting to Arduino on {self.port} @ {self.baudrate} baud...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # Flush buffers
            time.sleep(2)  # Wait for Arduino to initialize
            self.serial_conn.flushInput()
            self.serial_conn.flushOutput()
            
            self.is_connected = True
            self.retry_delay = 1  # Reset retry delay on success
            logger.info(f"✅ Connected to Arduino on {self.port}")
            return True
        
        except serial.SerialException as e:
            self.is_connected = False
            logger.warning(f"❌ Failed to connect: {e}")
            return False
        except Exception as e:
            self.is_connected = False
            logger.error(f"❌ Unexpected error during connection: {e}")
            return False
    
    def _reconnect_with_backoff(self) -> bool:
        """
        Attempt reconnection with exponential backoff.
        
        Returns:
            True if reconnected, False if still disconnected
        """
        if self.is_connected:
            return True
        
        logger.warning(f"⏳ Retrying in {self.retry_delay}s (backoff)...")
        time.sleep(self.retry_delay)
        
        if self._connect():
            return True
        else:
            # Increase delay exponentially, cap at max
            self.retry_delay = min(self.retry_delay * 2, self.max_retry_delay)
            return False
    
    def _parse_csv_line(self, line: str) -> Optional[Dict[str, float]]:
        """
        Parse CSV line from Arduino.
        
        Expected format: SENSOR1,SENSOR2,SENSOR3,...
        (no header, just comma-separated values)
        
        Args:
            line: CSV string from serial
            
        Returns:
            Dict mapping sensor names to values, or None if parse fails
        """
        try:
            line = line.strip()
            if not line:
                return None
            
            values = [float(v.strip()) for v in line.split(',')]
            
            if len(values) != len(self.sensor_cols):
                logger.warning(
                    f"⚠️ Expected {len(self.sensor_cols)} values, "
                    f"got {len(values)}: {line}"
                )
                return None
            
            return {col: val for col, val in zip(self.sensor_cols, values)}
        
        except ValueError as e:
            logger.warning(f"⚠️ Failed to parse line: {line} ({e})")
            return None
    
    def read_sample(self) -> Optional[Tuple[Dict[str, float], str]]:
        """
        Read one sensor sample from Arduino.
        
        Returns:
            Tuple of (sensor_data_dict, iso_timestamp_str) or None if read fails
        """
        # Try to reconnect if disconnected
        if not self.is_connected:
            if not self._reconnect_with_backoff():
                return None
        
        try:
            # Attempt to read line
            if self.serial_conn is None or not self.serial_conn.is_open:
                self.is_connected = False
                return None
            
            line = self.serial_conn.readline().decode('utf-8')
            
            if not line:
                # Timeout or empty read
                time_since_last = time.time() - self.last_successful_read_time
                if time_since_last > self.heartbeat_interval * 3:
                    logger.warning(
                        f"⚠️ No data for {time_since_last:.1f}s (heartbeat lost?)"
                    )
                return None
            
            # Parse the line
            sensor_data = self._parse_csv_line(line)
            if sensor_data is None:
                self.total_errors += 1
                return None
            
            # Record successful read
            self.last_successful_read_time = time.time()
            self.total_reads += 1
            
            # Generate ISO timestamp
            iso_timestamp = datetime.now().isoformat()
            
            logger.debug(f"📊 Read sample #{self.total_reads}: {sensor_data}")
            
            return (sensor_data, iso_timestamp)
        
        except serial.SerialException:
            self.is_connected = False
            logger.warning("❌ Serial connection lost (will auto-reconnect)")
            self.total_errors += 1
            return None
        
        except Exception as e:
            logger.error(f"❌ Unexpected error reading sample: {e}")
            self.total_errors += 1
            return None
    
    def close(self):
        """Close serial connection."""
        try:
            if self.serial_conn is not None and self.serial_conn.is_open:
                self.serial_conn.close()
                self.is_connected = False
                logger.info("🔌 Serial connection closed")
        except Exception as e:
            logger.error(f"Error closing serial connection: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Return connection statistics."""
        return {
            "is_connected": self.is_connected,
            "port": self.port,
            "total_reads": self.total_reads,
            "total_errors": self.total_errors,
            "error_rate": (
                self.total_errors / (self.total_reads + self.total_errors)
                if (self.total_reads + self.total_errors) > 0 else 0
            ),
            "seconds_since_last_read": time.time() - self.last_successful_read_time,
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
