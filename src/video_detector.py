"""
Video detection module for weapon detection system.
Detects weapons in video files using YOLO.
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
    is_valid_video_file
)
from src.alert_service import AlertService


class VideoDetector:
    """
    Detects weapons in video files.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize video detector.
        
        Args:
            model_path: Optional custom model path
        """
        self.logger = setup_logger("VideoDetector")
        self.alert_service = AlertService()
        
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
        video_path: Path,
        confidence_threshold: Optional[float] = None,
        save_output: bool = True,
        show_display: bool = False,
        frame_skip: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect weapons in a video file.
        
        Args:
            video_path: Path to input video
            confidence_threshold: Optional custom confidence threshold
            save_output: Whether to save annotated video
            show_display: Whether to display video while processing
            frame_skip: Process every Nth frame (None = use settings)
            
        Returns:
            Dictionary containing detection results
        """
        # Validate input
        if not is_valid_video_file(video_path):
            raise ValueError(f"Invalid video file: {video_path}")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(
            f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames"
        )
        
        # Set up video writer if saving
        video_writer = None
        output_path = None
        
        if save_output and Settings.SAVE_ANNOTATED_VIDEOS:
            output_path = self._create_output_path(video_path)
            fourcc = cv2.VideoWriter_fourcc(*Settings.OUTPUT_VIDEO_CODEC)
            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                Settings.OUTPUT_VIDEO_FPS,
                (width, height)
            )
        
        # Processing variables
        frame_skip = frame_skip or Settings.VIDEO_FRAME_SKIP
        conf = confidence_threshold or Settings.CONFIDENCE_THRESHOLD
        
        frame_count = 0
        processed_count = 0
        total_detections = 0
        weapon_detected_frames = 0
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    if video_writer:
                        video_writer.write(frame)
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
                    weapon_detected_frames += 1
                    
                    # Draw bounding boxes
                    annotated_frame = draw_bounding_boxes(annotated_frame, detections)
                    
                    # Draw alert banner
                    annotated_frame = draw_alert_banner(annotated_frame)
                    
                    # Trigger alerts
                    self.alert_service.trigger_alert(
                        detections,
                        source=f"video: {video_path.name} (frame {frame_count})",
                        frame=annotated_frame
                    )
                
                # Draw FPS
                current_fps = processed_count / (time.time() - start_time)
                annotated_frame = draw_fps(annotated_frame, current_fps)
                
                # Save frame
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Display frame
                if show_display:
                    cv2.imshow("Weapon Detection - Video", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("User stopped video processing")
                        break
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(
                        f"Progress: {progress:.1f}% "
                        f"({frame_count}/{total_frames} frames)"
                    )
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if show_display:
                cv2.destroyAllWindows()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_fps = processed_count / processing_time if processing_time > 0 else 0
        
        self.logger.info(
            f"Processing complete: {processed_count} frames in {processing_time:.1f}s "
            f"({avg_fps:.1f} FPS)"
        )
        
        if output_path:
            self.logger.info(f"Saved annotated video: {output_path}")
        
        return {
            "video_path": str(video_path),
            "weapon_detected": weapon_detected_frames > 0,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "frames_with_detections": weapon_detected_frames,
            "total_detections": total_detections,
            "processing_time": processing_time,
            "avg_fps": avg_fps,
            "output_path": str(output_path) if output_path else None
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
    
    def _create_output_path(self, video_path: Path) -> Path:
        """
        Create output path for annotated video.
        
        Args:
            video_path: Input video path
            
        Returns:
            Output video path
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_path.stem}_detected_{timestamp}.mp4"
        return Settings.OUTPUTS_DIR / filename


def detect_from_video(
    video_path: str,
    confidence: float = None,
    show: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to detect weapons in a video.
    
    Args:
        video_path: Path to video file (string)
        confidence: Optional confidence threshold (0.0 to 1.0)
        show: Whether to display the video
        
    Returns:
        Detection results dictionary
        
    Example:
        >>> result = detect_from_video("video.mp4")
        >>> print(f"Detected weapons in {result['frames_with_detections']} frames")
    """
    detector = VideoDetector()
    return detector.detect(
        Path(video_path),
        confidence_threshold=confidence,
        show_display=show
    )
