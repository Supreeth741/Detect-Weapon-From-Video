"""
Alert service for weapon detection system.
Handles multiple alert channels with rate limiting.
"""

import time
import winsound
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from plyer import notification

from config.settings import Settings
from src.utils import setup_logger, save_image, format_detection_log


class AlertService:
    """
    Manages alerts for weapon detection events.
    Supports console, sound, desktop notifications, and logging.
    """
    
    def __init__(self):
        """Initialize alert service."""
        self.logger = setup_logger("AlertService")
        self.last_alert_time = 0.0
        self.alert_count = 0
        self.detection_history: List[Dict[str, Any]] = []
    
    def should_trigger_alert(self) -> bool:
        """
        Check if enough time has passed since last alert (rate limiting).
        
        Returns:
            True if alert should be triggered
        """
        if not Settings.ALERT_ENABLED:
            return False
        
        current_time = time.time()
        time_since_last_alert = current_time - self.last_alert_time
        
        return time_since_last_alert >= Settings.ALERT_COOLDOWN
    
    def trigger_alert(
        self,
        detections: List[Dict[str, Any]],
        source: str,
        frame: Optional[Any] = None
    ) -> None:
        """
        Trigger all enabled alert channels.
        
        Args:
            detections: List of weapon detections
            source: Source of detection (image/video/camera)
            frame: Optional frame to save
        """
        if not self.should_trigger_alert():
            return
        
        # Update alert tracking
        self.last_alert_time = time.time()
        self.alert_count += 1
        
        # Record detection
        detection_record = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "detections": detections,
            "alert_number": self.alert_count
        }
        self.detection_history.append(detection_record)
        
        # Trigger all alert channels
        self._console_alert(detections, source)
        
        if Settings.ALERT_SOUND_ENABLED:
            self._sound_alert()
        
        if Settings.ALERT_NOTIFICATION_ENABLED:
            self._desktop_notification(detections, source)
        
        if Settings.ALERT_LOG_ENABLED:
            self._log_alert(detections, source)
        
        if Settings.ALERT_SAVE_FRAMES and frame is not None:
            self._save_detection_frame(frame, detections)
    
    def _console_alert(self, detections: List[Dict[str, Any]], source: str) -> None:
        """
        Print alert to console.
        
        Args:
            detections: List of detections
            source: Detection source
        """
        print("\n" + "=" * 60)
        print("⚠️  ALERT: WEAPON DETECTED! ⚠️")
        print("=" * 60)
        print(f"Source: {source}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of detections: {len(detections)}")
        print("\nDetected weapons:")
        
        for idx, detection in enumerate(detections, 1):
            class_name = detection['class_name']
            confidence = detection['confidence']
            print(f"  {idx}. {class_name} (confidence: {confidence:.2%})")
        
        print("=" * 60 + "\n")
    
    def _sound_alert(self) -> None:
        """
        Play alert sound (Windows system beep).
        """
        try:
            # Play Windows system beep (frequency, duration in ms)
            winsound.Beep(1000, 500)  # 1000 Hz for 500ms
        except Exception as e:
            self.logger.warning(f"Failed to play alert sound: {e}")
    
    def _desktop_notification(self, detections: List[Dict[str, Any]], source: str) -> None:
        """
        Show desktop notification (Windows toast).
        
        Args:
            detections: List of detections
            source: Detection source
        """
        try:
            weapon_types = ", ".join(set(d['class_name'] for d in detections))
            message = f"Detected: {weapon_types}\nSource: {source}"
            
            notification.notify(
                title="⚠️ Weapon Detected!",
                message=message,
                app_name="Weapon Detector",
                timeout=10  # Notification duration in seconds
            )
        except Exception as e:
            self.logger.warning(f"Failed to show desktop notification: {e}")
    
    def _log_alert(self, detections: List[Dict[str, Any]], source: str) -> None:
        """
        Log detection to file.
        
        Args:
            detections: List of detections
            source: Detection source
        """
        log_message = format_detection_log(detections, source)
        self.logger.warning(log_message)
    
    def _save_detection_frame(self, frame: Any, detections: List[Dict[str, Any]]) -> None:
        """
        Save frame with detections to disk.
        
        Args:
            frame: Frame to save
            detections: List of detections
        """
        try:
            saved_path = save_image(frame, Settings.OUTPUTS_DIR, prefix="alert")
            self.logger.info(f"Saved detection frame: {saved_path}")
        except Exception as e:
            self.logger.error(f"Failed to save detection frame: {e}")
    
    def save_detection_log(self, output_file: Optional[Path] = None) -> Path:
        """
        Save detection history to JSON file.
        
        Args:
            output_file: Optional custom output file path
            
        Returns:
            Path to saved log file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Settings.LOGS_DIR / f"detections_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = []
        for record in self.detection_history:
            serializable_record = record.copy()
            for detection in serializable_record['detections']:
                if 'bbox' in detection:
                    detection['bbox'] = [float(x) for x in detection['bbox']]
            serializable_history.append(serializable_record)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Saved detection log to: {output_file}")
        return output_file
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        return {
            "total_alerts": self.alert_count,
            "total_detections": sum(
                len(record['detections']) for record in self.detection_history
            ),
            "detection_history_count": len(self.detection_history),
            "last_alert_time": datetime.fromtimestamp(self.last_alert_time).isoformat()
            if self.last_alert_time > 0 else None
        }
    
    def reset(self) -> None:
        """Reset alert service state."""
        self.last_alert_time = 0.0
        self.alert_count = 0
        self.detection_history.clear()
        self.logger.info("Alert service reset")
