"""
Configuration settings for weapon detection system.
Centralizes all configuration parameters for easy modification.
"""

from pathlib import Path
from typing import Dict, List


class Settings:
    """Central configuration for weapon detection system."""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    OUTPUTS_DIR = DATA_DIR / "outputs"
    
    # Model configuration
    MODEL_NAME = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
    MODEL_PATH = MODELS_DIR / MODEL_NAME
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence score (0.0 to 1.0)
    IOU_THRESHOLD = 0.45  # Intersection over Union threshold for NMS
    
    # Weapon classes to detect
    # Note: Standard COCO models don't include weapons
    # You'll need a custom-trained model for weapon detection
    WEAPON_CLASSES = {
        "knife": 0,
        "gun": 1,
        "pistol": 2,
        "rifle": 3,
        "weapon": 4,
    }
    
    # Visual settings
    BBOX_COLOR = (0, 0, 255)  # Red color in BGR format
    BBOX_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_THICKNESS = 2
    FONT_SCALE = 0.6
    
    # Camera settings
    CAMERA_ID = 0  # Default camera (0 is usually webcam)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    FRAME_SKIP = 1  # Process every Nth frame (1 = no skip)
    
    # Video processing settings
    VIDEO_FRAME_SKIP = 2  # Process every 2nd frame for videos (faster)
    OUTPUT_VIDEO_FPS = 30
    OUTPUT_VIDEO_CODEC = "mp4v"  # FOURCC codec
    
    # Alert settings
    ALERT_ENABLED = True
    ALERT_COOLDOWN = 5.0  # Minimum seconds between alerts (prevents spam)
    ALERT_SOUND_ENABLED = True
    ALERT_NOTIFICATION_ENABLED = True  # Desktop notifications
    ALERT_LOG_ENABLED = True
    ALERT_SAVE_FRAMES = True  # Save frames with detections
    
    # Logging settings
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = LOGS_DIR / "detections.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Output settings
    SAVE_ANNOTATED_IMAGES = True
    SAVE_ANNOTATED_VIDEOS = True
    OUTPUT_IMAGE_FORMAT = "jpg"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Get the path to the YOLO model, with fallback options."""
        if cls.MODEL_PATH.exists():
            return cls.MODEL_PATH
        
        # If custom model doesn't exist, fallback to default YOLOv8n
        return cls.MODELS_DIR / "yolov8n.pt"
    
    @classmethod
    def update_confidence(cls, confidence: float) -> None:
        """Update confidence threshold dynamically."""
        if 0.0 <= confidence <= 1.0:
            cls.CONFIDENCE_THRESHOLD = confidence
        else:
            raise ValueError("Confidence must be between 0.0 and 1.0")


# Initialize directories on import
Settings.ensure_directories()
