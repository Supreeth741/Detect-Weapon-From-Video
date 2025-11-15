"""
Image detection module for weapon detection system.
Detects weapons in static images using YOLO.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from config.settings import Settings
from src.utils import (
    setup_logger,
    draw_bounding_boxes,
    draw_alert_banner,
    save_image,
    load_image,
    is_valid_image_file
)
from src.alert_service import AlertService


class ImageDetector:
    """
    Detects weapons in static images.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize image detector.
        
        Args:
            model_path: Optional custom model path
        """
        self.logger = setup_logger("ImageDetector")
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
        image_path: Path,
        confidence_threshold: Optional[float] = None,
        save_output: bool = True,
        show_display: bool = False
    ) -> Dict[str, Any]:
        """
        Detect weapons in a single image.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Optional custom confidence threshold
            save_output: Whether to save annotated image
            show_display: Whether to display the image
            
        Returns:
            Dictionary containing detection results
        """
        # Validate input
        if not is_valid_image_file(image_path):
            raise ValueError(f"Invalid image file: {image_path}")
        
        # Load image
        self.logger.info(f"Processing image: {image_path}")
        image = load_image(image_path)
        
        # Run detection
        results = self._run_detection(image, confidence_threshold)
        
        # Process results
        detections = self._process_results(results)
        
        # Handle detections
        weapon_detected = len(detections) > 0
        annotated_image = image.copy()
        
        if weapon_detected:
            # Draw bounding boxes
            annotated_image = draw_bounding_boxes(annotated_image, detections)
            
            # Draw alert banner
            annotated_image = draw_alert_banner(annotated_image)
            
            # Trigger alerts
            self.alert_service.trigger_alert(
                detections,
                source=f"image: {image_path.name}",
                frame=annotated_image
            )
        else:
            self.logger.info("No weapons detected")
        
        # Save output
        output_path = None
        if save_output and Settings.SAVE_ANNOTATED_IMAGES:
            output_path = save_image(
                annotated_image,
                Settings.OUTPUTS_DIR,
                prefix=image_path.stem
            )
            self.logger.info(f"Saved annotated image: {output_path}")
        
        # Display image
        if show_display:
            self._display_image(annotated_image, image_path.name)
        
        return {
            "image_path": str(image_path),
            "weapon_detected": weapon_detected,
            "num_detections": len(detections),
            "detections": detections,
            "output_path": str(output_path) if output_path else None
        }
    
    def _run_detection(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> Any:
        """
        Run YOLO detection on image.
        
        Args:
            image: Input image
            confidence_threshold: Optional confidence threshold
            
        Returns:
            YOLO results object
        """
        conf = confidence_threshold or Settings.CONFIDENCE_THRESHOLD
        
        results = self.model.predict(
            image,
            conf=conf,
            iou=Settings.IOU_THRESHOLD,
            verbose=False
        )
        
        return results[0] if results else None
    
    def _process_results(self, results: Any) -> List[Dict[str, Any]]:
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
            # Extract box information
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            detection = {
                "bbox": bbox.tolist(),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            }
            
            detections.append(detection)
        
        return detections
    
    def _display_image(self, image: np.ndarray, window_name: str) -> None:
        """
        Display image in a window.
        
        Args:
            image: Image to display
            window_name: Window title
        """
        cv2.imshow(window_name, image)
        self.logger.info("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_batch(
        self,
        image_paths: List[Path],
        confidence_threshold: Optional[float] = None,
        save_output: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect weapons in multiple images.
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Optional confidence threshold
            save_output: Whether to save outputs
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.detect(
                    image_path,
                    confidence_threshold=confidence_threshold,
                    save_output=save_output,
                    show_display=False
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "error": str(e)
                })
        
        return results


def detect_from_image(
    image_path: str,
    confidence: float = None,
    show: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to detect weapons in an image.
    
    Args:
        image_path: Path to image file (string)
        confidence: Optional confidence threshold (0.0 to 1.0)
        show: Whether to display the image
        
    Returns:
        Detection results dictionary
        
    Example:
        >>> result = detect_from_image("photo.jpg")
        >>> if result['weapon_detected']:
        ...     print(f"Found {result['num_detections']} weapons!")
    """
    detector = ImageDetector()
    return detector.detect(
        Path(image_path),
        confidence_threshold=confidence,
        show_display=show
    )
