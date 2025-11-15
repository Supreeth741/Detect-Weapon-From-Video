"""
Utility functions for weapon detection system.
Provides helper functions for drawing, logging, and file operations.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

from config.settings import Settings


def setup_logger(name: str = "weapon_detector") -> logging.Logger:
    """
    Set up and configure logger for the application.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Settings.LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(Settings.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(Settings.LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image (numpy array)
        detections: List of detection dictionaries with keys:
                   'bbox', 'confidence', 'class_name'
        show_confidence: Whether to show confidence scores
        
    Returns:
        Image with drawn bounding boxes
    """
    annotated_image = image.copy()
    
    for detection in detections:
        # Extract detection info
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Convert bbox to integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            Settings.BBOX_COLOR,
            Settings.BBOX_THICKNESS
        )
        
        # Prepare label text
        if show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            Settings.FONT_SCALE,
            Settings.TEXT_THICKNESS
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            Settings.BBOX_COLOR,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            Settings.FONT_SCALE,
            Settings.TEXT_COLOR,
            Settings.TEXT_THICKNESS,
            cv2.LINE_AA
        )
    
    return annotated_image


def draw_alert_banner(image: np.ndarray, text: str = "⚠️ WEAPON DETECTED!") -> np.ndarray:
    """
    Draw a prominent alert banner on the image.
    
    Args:
        image: Input image
        text: Alert text to display
        
    Returns:
        Image with alert banner
    """
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    # Banner dimensions
    banner_height = 60
    banner_color = (0, 0, 255)  # Red
    text_color = (255, 255, 255)  # White
    
    # Draw banner background
    cv2.rectangle(
        annotated_image,
        (0, 0),
        (width, banner_height),
        banner_color,
        -1
    )
    
    # Calculate text position (centered)
    font_scale = 1.0
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )
    
    text_x = (width - text_width) // 2
    text_y = (banner_height + text_height) // 2
    
    # Draw text
    cv2.putText(
        annotated_image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return annotated_image


def save_image(image: np.ndarray, output_path: Path, prefix: str = "detection") -> Path:
    """
    Save image with a timestamped filename.
    
    Args:
        image: Image to save
        output_path: Directory to save the image
        prefix: Filename prefix
        
    Returns:
        Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.{Settings.OUTPUT_IMAGE_FORMAT}"
    full_path = output_path / filename
    
    cv2.imwrite(str(full_path), image)
    return full_path


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image cannot be loaded
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def resize_frame(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame


def format_detection_log(detections: List[Dict[str, Any]], source: str) -> str:
    """
    Format detection information for logging.
    
    Args:
        detections: List of detections
        source: Source of detection (image/video/camera)
        
    Returns:
        Formatted log string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [f"[{timestamp}] Weapon Detection Alert - Source: {source}"]
    
    for idx, detection in enumerate(detections, 1):
        class_name = detection['class_name']
        confidence = detection['confidence']
        bbox = detection['bbox']
        log_lines.append(
            f"  Detection {idx}: {class_name} "
            f"(confidence: {confidence:.2%}, bbox: {bbox})"
        )
    
    return "\n".join(log_lines)


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate frames per second.
    
    Args:
        start_time: Start timestamp
        frame_count: Number of frames processed
        
    Returns:
        FPS value
    """
    import time
    elapsed_time = time.time() - start_time
    
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS counter on image.
    
    Args:
        image: Input image
        fps: FPS value
        
    Returns:
        Image with FPS counter
    """
    annotated_image = image.copy()
    fps_text = f"FPS: {fps:.1f}"
    
    cv2.putText(
        annotated_image,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Green
        2,
        cv2.LINE_AA
    )
    
    return annotated_image


def is_valid_video_file(file_path: Path) -> bool:
    """
    Check if file is a valid video file.
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if valid video file
    """
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return file_path.suffix.lower() in valid_extensions


def is_valid_image_file(file_path: Path) -> bool:
    """
    Check if file is a valid image file.
    
    Args:
        file_path: Path to image file
        
    Returns:
        True if valid image file
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return file_path.suffix.lower() in valid_extensions
