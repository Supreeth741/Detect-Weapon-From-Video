"""
Camera detection module for weapon detection system.
Detects weapons in real-time from webcam or camera feed.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO

from config.settings import Settings
from src.utils import (
    setup_logger,
    draw_bounding_boxes,
    draw_alert_banner,
    draw_fps,
    save_image
)
from src.alert_service import AlertService


class CameraDetector:
    """
    Detects weapons in real-time from camera feed.
    """
    
    def __init__(self, model_path: Optional[Path] = None, camera_id: int = None):
        """
        Initialize camera detector.
        
        Args:
            model_path: Optional custom model path
            camera_id: Camera device ID (default from settings)
        """
        self.logger = setup_logger("CameraDetector")
        self.alert_service = AlertService()
        self.camera_id = camera_id or Settings.CAMERA_ID
        
        # Load YOLO model
        if model_path is None:
            model_path = Settings.get_model_path()
        
        self.logger.info(f"Loading YOLO model: {model_path}")
        
        try:
            self.model = YOLO(str(model_path))
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(
        self,
        confidence_threshold: Optional[float] = None,
        frame_skip: Optional[int] = None,
        max_runtime: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Start real-time weapon detection from camera.
        
        Args:
            confidence_threshold: Optional custom confidence threshold
            frame_skip: Process every Nth frame (None = use settings)
            max_runtime: Optional maximum runtime in seconds
            
        Returns:
            Dictionary containing detection statistics
        """
        self.logger.info(f"Starting camera detection (device {self.camera_id})")
        
        # Open camera with DirectShow backend (Windows)
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {self.camera_id}. "
                "Check camera permissions in Windows Settings."
            )
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Settings.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Settings.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Settings.CAMERA_FPS)
        
        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        self.logger.info(
            f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS"
        )
        
        # Processing variables
        frame_skip = frame_skip or Settings.FRAME_SKIP
        conf = confidence_threshold or Settings.CONFIDENCE_THRESHOLD
        
        frame_count = 0
        processed_count = 0
        total_detections = 0
        frames_with_detections = 0
        
        start_time = time.time()
        paused = False
        
        self.logger.info("Camera detection started. Press 'q' to quit, 'p' to pause, 's' to screenshot")
        
        try:
            while True:
                # Check max runtime
                if max_runtime and (time.time() - start_time) > max_runtime:
                    self.logger.info("Max runtime reached")
                    break
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Handle pause
                if paused:
                    cv2.imshow("Weapon Detection - Camera (PAUSED)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        paused = False
                        self.logger.info("Resumed")
                    elif key == ord('q'):
                        break
                    continue
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    continue
                
                processed_count += 1
                
                # Run detection
                results = self.model.predict(
                    frame,
                    conf=conf,
                    iou=Settings.IOU_THRESHOLD,
                    verbose=False
                )
                
                # Process results
                detections = self._process_results(results[0] if results else None)
                
                # Annotate frame
                annotated_frame = frame.copy()
                
                if detections:
                    total_detections += len(detections)
                    frames_with_detections += 1
                    
                    # Draw bounding boxes
                    annotated_frame = draw_bounding_boxes(annotated_frame, detections)
                    
                    # Draw alert banner
                    annotated_frame = draw_alert_banner(annotated_frame)
                    
                    # Trigger alerts
                    self.alert_service.trigger_alert(
                        detections,
                        source=f"camera {self.camera_id}",
                        frame=annotated_frame
                    )
                
                # Draw FPS
                current_fps = processed_count / (time.time() - start_time)
                annotated_frame = draw_fps(annotated_frame, current_fps)
                
                # Add instructions
                self._draw_instructions(annotated_frame)
                
                # Display frame
                cv2.imshow("Weapon Detection - Camera", annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("User quit")
                    break
                elif key == ord('p'):
                    paused = True
                    self.logger.info("Paused")
                elif key == ord('s'):
                    # Take screenshot
                    screenshot_path = save_image(
                        annotated_frame,
                        Settings.OUTPUTS_DIR,
                        prefix="camera_screenshot"
                    )
                    self.logger.info(f"Screenshot saved: {screenshot_path}")
                    print(f"📸 Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_fps = processed_count / processing_time if processing_time > 0 else 0
        
        self.logger.info(
            f"Detection stopped: {processed_count} frames processed in {processing_time:.1f}s "
            f"({avg_fps:.1f} FPS)"
        )
        
        return {
            "camera_id": self.camera_id,
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "frames_with_detections": frames_with_detections,
            "total_detections": total_detections,
            "processing_time": processing_time,
            "avg_fps": avg_fps,
            "weapon_detected": frames_with_detections > 0
        }
    
    def _process_results(self, results: Any) -> list:
        """
        Process YOLO results into detection dictionaries.
        
        Args:
            results: YOLO results object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if results is None or results.boxes is None:
            return detections
        
        boxes = results.boxes
        
        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            detections.append({
                "bbox": bbox.tolist(),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            })
        
        return detections
    
    def _draw_instructions(self, frame: np.ndarray) -> None:
        """
        Draw keyboard instructions on frame.
        
        Args:
            frame: Frame to draw on (modified in place)
        """
        instructions = [
            "Q: Quit",
            "P: Pause",
            "S: Screenshot"
        ]
        
        y_offset = frame.shape[0] - 20
        x_offset = 10
        
        for instruction in reversed(instructions):
            cv2.putText(
                frame,
                instruction,
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset -= 20


def detect_from_camera(
    camera_id: int = 0,
    confidence: float = None,
    max_runtime: int = None
) -> Dict[str, Any]:
    """
    Convenience function to detect weapons from camera.
    
    Args:
        camera_id: Camera device ID (default 0)
        confidence: Optional confidence threshold (0.0 to 1.0)
        max_runtime: Optional maximum runtime in seconds
        
    Returns:
        Detection statistics dictionary
        
    Example:
        >>> result = detect_from_camera()
        >>> print(f"Detected weapons in {result['frames_with_detections']} frames")
    """
    detector = CameraDetector(camera_id=camera_id)
    return detector.detect(
        confidence_threshold=confidence,
        max_runtime=max_runtime
    )
