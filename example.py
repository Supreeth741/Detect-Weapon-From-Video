"""
Example: Simple usage of weapon detection system
"""

from src.image_detector import detect_from_image
from src.video_detector import detect_from_video
from src.camera_detector import detect_from_camera


def example_image_detection():
    """Example: Detect weapons in an image."""
    print("=" * 60)
    print("Example 1: Image Detection")
    print("=" * 60)
    
    # Detect from image
    result = detect_from_image(
        "data/images/sample.jpg",
        confidence=0.5,
        show=True
    )
    
    print(f"Weapon detected: {result['weapon_detected']}")
    print(f"Number of detections: {result['num_detections']}")
    
    if result['detections']:
        print("\nDetected weapons:")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class_name']} ({det['confidence']:.2%})")


def example_video_detection():
    """Example: Detect weapons in a video."""
    print("\n" + "=" * 60)
    print("Example 2: Video Detection")
    print("=" * 60)
    
    # Detect from video
    result = detect_from_video(
        "data/videos/sample.mp4",
        confidence=0.5,
        show=False
    )
    
    print(f"Weapon detected: {result['weapon_detected']}")
    print(f"Frames with detections: {result['frames_with_detections']}")
    print(f"Total detections: {result['total_detections']}")
    print(f"Average FPS: {result['avg_fps']:.1f}")


def example_camera_detection():
    """Example: Real-time camera detection."""
    print("\n" + "=" * 60)
    print("Example 3: Live Camera Detection")
    print("=" * 60)
    
    print("Starting camera... Press 'q' to quit\n")
    
    # Detect from camera (30 second limit for demo)
    result = detect_from_camera(
        camera_id=0,
        confidence=0.5,
        max_runtime=30
    )
    
    print(f"\nSession complete!")
    print(f"Weapon detected: {result['weapon_detected']}")
    print(f"Frames with detections: {result['frames_with_detections']}")
    print(f"Average FPS: {result['avg_fps']:.1f}")


if __name__ == "__main__":
    print("Weapon Detection System - Examples\n")
    
    # Uncomment the examples you want to run:
    
    # example_image_detection()
    # example_video_detection()
    # example_camera_detection()
    
    print("\nNote: Uncomment the example functions you want to run!")
    print("Make sure you have sample files in data/images/ and data/videos/")
