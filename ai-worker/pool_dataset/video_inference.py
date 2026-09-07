#!/usr/bin/env python3
"""
YOLOv8 Video Inference Script for Pool Balls Detection
=====================================================

This script runs inference on a video using a trained YOLOv8 model
for pool balls detection and saves the results to a new video.

Usage:
    python video_inference.py --model MODEL_PATH --input INPUT_VIDEO --output OUTPUT_VIDEO

Example:
    python video_inference.py --model runs/train/pool_balls_yolo8m_20251022_00402022/weights/best.pt --input output.mp4 --output detected_pool_balls.mp4
"""

import os
import argparse
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoInference:
    """Video Inference Class for Pool Balls Detection"""
    
    def __init__(self, model_path, confidence=0.5, device=None):
        """
        Initialize the video inference
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
            confidence (float): Confidence threshold for detections
            device (str): Device to use for inference ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the trained model
        self.load_model()
        
        # Pool ball colors for visualization (16 colors for 16 ball numbers)
        self.colors = self.generate_colors(16)
        
        # Class names for pool balls (0-15)
        self.class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def load_model(self):
        """Load the trained YOLOv8 model"""
        try:
            logger.info(f"Loading trained model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            # Generate HSV colors and convert to BGR
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append([int(c) for c in color])
        return colors
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
            confidences = results[0].boxes.conf.cpu().numpy()  # Get confidences
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if conf >= self.confidence:
                    # Extract coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get color for this class
                    color = self.colors[class_id % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    label = f"{self.class_names[class_id]}: {conf:.2f}"
                    
                    # Get text size for background
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), color, -1)
                    
                    # Draw text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, input_path, output_path, show_preview=False):
        """
        Process video with pool balls detection
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            show_preview (bool): Whether to show real-time preview
        """
        # Verify input video exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties:")
        logger.info(f"  - Resolution: {width}x{height}")
        logger.info(f"  - FPS: {fps}")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - Duration: {total_frames/fps:.2f} seconds")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        
        try:
            logger.info("Starting video processing...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                results = self.model(frame, conf=self.confidence, device=self.device)
                
                # Count detections in this frame
                if results[0].boxes is not None:
                    frame_detections = len(results[0].boxes)
                    detection_count += frame_detections
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, results)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Show preview if requested
                if show_preview:
                    cv2.imshow('Pool Balls Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing stopped by user")
                        break
                
                # Progress update
                if frame_count % 30 == 0:  # Update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            logger.info("Video processing completed!")
            logger.info(f"Total frames processed: {frame_count}")
            logger.info(f"Total detections: {detection_count}")
            logger.info(f"Average detections per frame: {detection_count/frame_count:.2f}")
            
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            raise
        finally:
            # Clean up
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
    
    def process_image(self, input_path, output_path):
        """
        Process single image with pool balls detection
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to output image
        """
        # Verify input image exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        logger.info(f"Processing image: {input_path}")
        
        # Run inference
        results = self.model(image, conf=self.confidence, device=self.device)
        
        # Draw detections
        annotated_image = self.draw_detections(image, results)
        
        # Add detection count
        detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
        info_text = f"Detections: {detection_count}"
        cv2.putText(annotated_image, info_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Result saved to: {output_path}")
        logger.info(f"Detections found: {detection_count}")


def main():
    """Main function to run video inference"""
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on video for pool balls detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLOv8 model (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output video file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--preview', action='store_true',
                       help='Show real-time preview during processing')
    parser.add_argument('--image', action='store_true',
                       help='Process as single image instead of video')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        logger.info("Initializing video inference...")
        inference = VideoInference(
            model_path=args.model,
            confidence=args.confidence,
            device=args.device
        )
        
        # Process video or image
        if args.image:
            logger.info("Processing single image...")
            inference.process_image(args.input, args.output)
        else:
            logger.info("Processing video...")
            inference.process_video(args.input, args.output, show_preview=args.preview)
        
        logger.info("Inference completed successfully!")
        logger.info(f"Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()

