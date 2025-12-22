import pandas as pd

def generate_alert_for_segments(sensor_df, use_case, confidence_threshold=0.15):
    """Generate one alert per anomaly segment"""
    
    messages = {
        "door": "door left open",
        "co2": "unauthorized person detected", 
        "appliance": "unattended appliance",
        "abnormal_object": "object blocking hallway"
    }
    
    locations = {
        "door": "Entrance",
        "co2": "House",
        "appliance": "Kitchen", 
        "abnormal_object": "Hallway"
    }
    
    """Generate one alert per anomaly segment"""

    # Enforce unique index
    sensor_df.index = pd.RangeIndex(len(sensor_df))

    # Filtering
    anomalies = sensor_df[
        (sensor_df["Sensor_Status"] == "Anomaly") & 
        (sensor_df["Confidence"] > confidence_threshold)
    ].copy()
    
    if sensor_df.index.duplicated().any():
        print("⚠️ WARNING: Duplicate index detected in sensor_df")
    else:
        print("✅ Index is unique before filtering")


    if len(anomalies) == 0:
        return
    
    # Convert timestamps to datetime
    anomalies["Sensor_Timestamp"] = pd.to_datetime(anomalies["Sensor_Timestamp"])
    anomalies = anomalies.sort_values("Sensor_Timestamp")
    
    # Group into segments (gap > 5 minutes = new segment)
    segments = []
    current_segment = [anomalies.iloc[0]]
    
    for i in range(1, len(anomalies)):
        time_diff = (anomalies.iloc[i]["Sensor_Timestamp"] - 
                    current_segment[-1]["Sensor_Timestamp"]).total_seconds() / 60
        
        if time_diff <= 5:  # Same segment if within 5 minutes
            current_segment.append(anomalies.iloc[i])
        else:
            segments.append(current_segment)
            current_segment = [anomalies.iloc[i]]
    
    segments.append(current_segment)
    
    # Generate one alert per segment
    for segment in segments:
        start_time = segment[0]["Sensor_Timestamp"]
        max_confidence = max(row["Confidence"] for row in segment)
        
        time_str = start_time.strftime("%H:%M")
        location = locations.get(use_case, "Room")
        anomaly_type = messages.get(use_case, "anomaly detected")
        
        alert_msg = f"**{location}: {anomaly_type}, {max_confidence*100:.0f}% confidence – {time_str}**"
        print(f"🚨 ALERT: {alert_msg}")

import pandas as pd

def generate_alert_for_vision_fallback(vision_df, use_case, confidence_threshold=0.8):
    """
    Generate alerts for vision-based fallback anomalies only.
    Keeps logic separate from sensor-based confidence estimation.
    """
    messages = {
        "door": "door left open",
        "co2": "unauthorized person detected",
        "appliance": "unattended appliance",
        "abnormal_object": "object blocking hallway"
    }

    locations = {
        "door": "Entrance",
        "co2": "House",
        "appliance": "Kitchen",
        "abnormal_object": "Hallway"
    }

    # ✅ Check for and remove duplicate columns
    vision_df = vision_df.loc[:, ~vision_df.columns.duplicated()].copy()

    # ✅ Clean index to avoid reindex errors
    vision_df.reset_index(drop=True, inplace=True)

    # ✅ Confirm required columns exist
    required_cols = ["Sensor_Status", "Confidence", "Sensor_Timestamp"]
    missing = [col for col in required_cols if col not in vision_df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return

    # ✅ Filter to high-confidence vision anomalies
    anomalies = vision_df[
        (vision_df["Sensor_Status"] == "Anomaly") &
        (vision_df["Confidence"] > confidence_threshold)
    ].copy()

    if anomalies.empty:
        print("✅ No high-confidence vision anomalies to alert on.")
        return

    # Parse timestamps
    anomalies["Sensor_Timestamp"] = pd.to_datetime(anomalies["Sensor_Timestamp"], errors="coerce")
    anomalies = anomalies.sort_values("Sensor_Timestamp")

    # Segment by 5-minute gap
    segments = []
    current = [anomalies.iloc[0]]

    for i in range(1, len(anomalies)):
        time_diff = (anomalies.iloc[i]["Sensor_Timestamp"] - current[-1]["Sensor_Timestamp"]).total_seconds() / 60
        if time_diff <= 5:
            current.append(anomalies.iloc[i])
        else:
            segments.append(current)
            current = [anomalies.iloc[i]]

    segments.append(current)

    # Print alerts
    for segment in segments:
        start_time = segment[0]["Sensor_Timestamp"]
        max_conf = max(row["Confidence"] for row in segment)
        time_str = start_time.strftime("%H:%M")
        location = locations.get(use_case, "Room")
        message = messages.get(use_case, "anomaly detected")

        alert = f"**{location}: {message}, {max_conf * 100:.0f}% confidence – {time_str}**"
        print(f"🚨 ALERT: {alert}")
