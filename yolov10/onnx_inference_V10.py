import numpy as np
import cv2
import onnxruntime
import argparse
import time
from pathlib import Path

class YOLOInference:
    def __init__(self, model_path, labels_path=None, conf_thres=0.25, img_size=(640, 640)):
        """
        Initialize YOLO inference class
        Args:
            model_path: Path to ONNX model
            labels_path: Path to labels file (optional)
            conf_thres: Confidence threshold for filtering detections
            img_size: Input image size (height, width)
        """
        self.conf_thres = conf_thres
        self.img_size = img_size
        
        # Load labels
        self.labels = []
        if labels_path:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        # Initialize ONNX Runtime
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, img):
        """
        Preprocess image for inference
        Args:
            img: Input image in BGR format
        Returns:
            Preprocessed image
        """
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img

    def postprocess(self, num_dets, det_boxes, det_scores, det_classes, orig_img):
        """
        Postprocess detections
        Args:
            num_dets: Number of detections per image
            det_boxes: Detection boxes
            det_scores: Detection scores
            det_classes: Detection classes
            orig_img: Original image
        Returns:
            processed_detections: List of detections with scaled coordinates
        """
        processed_detections = []
        
        # Get image dimensions for scaling
        img_height, img_width = orig_img.shape[:2]
        scale_x = img_width / self.img_size[1]
        scale_y = img_height / self.img_size[0]
        
        # Process valid detections
        for idx in range(int(num_dets[0][0])):
            if det_scores[0][idx] > self.conf_thres:
                # Get scaled coordinates
                x1, y1, x2, y2 = det_boxes[0][idx]
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Get class details
                class_id = int(det_classes[0][idx])
                score = float(det_scores[0][idx])
                label = self.labels[class_id] if self.labels else str(class_id)
                
                processed_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': score,
                    'class_id': class_id,
                    'label': label
                })
        
        return processed_detections

    def draw_detections(self, img, detections):
        """
        Draw detections on image
        Args:
            img: Original image
            detections: List of processed detections
        Returns:
            img: Image with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['score']:.2f}"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return img

    def __call__(self, img):
        """
        Run inference on image
        Args:
            img: Input image in BGR format
        Returns:
            detections: List of processed detections
        """
        # Preprocess image
        input_img = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_img})
        num_dets, det_boxes, det_scores, det_classes = outputs
        
        # Postprocess detections
        detections = self.postprocess(num_dets, det_boxes, det_scores, det_classes, img)
        
        return detections

def process_video(model, video_source, output_path=None, show_display=True):
    """
    Process video from file or webcam
    Args:
        model: YOLOInference instance
        video_source: Video file path or camera index
        output_path: Path to save output video (optional)
        show_display: Whether to show live display
    """
    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {video_source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        start_time = time.time()
        detections = model(frame)
        inference_time = time.time() - start_time
        
        # Update statistics
        frame_count += 1
        total_time += inference_time
        
        # Draw detections and FPS
        result_frame = model.draw_detections(frame.copy(), detections)
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame if output path is provided
        if writer:
            writer.write(result_frame)
            
        # Show live display if requested
        if show_display:
            cv2.imshow('YOLO Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    
    # Print statistics
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.1f} seconds ({avg_fps:.1f} FPS)")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO ONNX Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--labels', type=str, help='Path to labels file')
    parser.add_argument('--source', type=str, required=True, 
                      help='Path to input image/video file, or camera index (e.g., 0 for webcam)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Path to output image/video')
    parser.add_argument('--no-display', action='store_true', help='Disable live display for video')
    args = parser.parse_args()

    # Initialize YOLOInference
    model = YOLOInference(
        model_path=args.model,
        labels_path=args.labels,
        conf_thres=args.conf_thres
    )

    # Check if source is an image file
    source_path = Path(args.source)
    if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        img = cv2.imread(args.source)
        if img is None:
            raise ValueError(f"Failed to read image: {args.source}")

        # Run inference
        start_time = time.time()
        detections = model(img)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")

        # Draw detections
        result_img = model.draw_detections(img.copy(), detections)

        # Save result
        output_path = args.output or 'output.jpg'
        cv2.imwrite(output_path, result_img)
        print(f"Results saved to {output_path}")

        # Print detections
        for det in detections:
            print(f"Detected {det['label']} with confidence {det['score']:.2f} at {det['bbox']}")
    
    else:
        # Process video/webcam
        try:
            # If source is a number string (e.g., "0"), convert to int for webcam
            video_source = int(args.source)
        except ValueError:
            # Otherwise, use the string path
            video_source = args.source
            
        process_video(
            model=model,
            video_source=video_source,
            output_path=args.output,
            show_display=not args.no_display
        )

if __name__ == '__main__':
    main()