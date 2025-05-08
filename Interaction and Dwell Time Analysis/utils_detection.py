"""
Object detection utilities using ONNX Runtime.
"""
import cv2
import numpy as np
import onnxruntime as ort

class ObjectDetector:
    """YOLO object detector using ONNX Runtime."""
    
    def __init__(self, model_path, providers, input_width=640, input_height=640):
        """
        Initialize the detector with the ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            input_width: Model input width
            input_height: Model input height
        """
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model details
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_width = input_width
        self.input_height = input_height
    
    def preprocess_image(self, img):
        """
        Preprocess the image for YOLO inference.
        
        Args:
            img: Input image (BGR format from OpenCV)
            
        Returns:
            tuple: (preprocessed blob, ratio, original image shape)
        """
        img_height, img_width = img.shape[:2]
        r = min(self.input_width/img_width, self.input_height/img_height)
        new_width, new_height = int(img_width * r), int(img_height * r)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width, :] = resized
        
        # Convert to RGB and normalize
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # NCHW format for ONNX
        blob = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]
        
        return blob, r, (img_width, img_height)
    
    def process_output(self, outputs, conf_threshold=0.3, nms_threshold=0.5, img_shape=None, ratio=1.0):
        """
        Process YOLO model output with NMS.
        
        Args:
            outputs: Model outputs
            conf_threshold: Confidence threshold
            nms_threshold: Non-maximum suppression threshold
            img_shape: Original image shape
            ratio: Scaling ratio from preprocessing
            
        Returns:
            list: Filtered detections with boxes in format [x1, y1, x2, y2]
        """
        output = outputs[0][0]
        boxes, confidences = [], []
        
        for idx in range(output.shape[1]):
            confidence = output[4, idx]
            if confidence >= conf_threshold:
                x, y, w, h = output[:4, idx]
                x1 = int((x - w / 2) / ratio)
                y1 = int((y - h / 2) / ratio)
                width, height = int(w / ratio), int(h / ratio)
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        detections = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'box': [x, y, x + w, y + h],  # Convert to [x1, y1, x2, y2] format
                    'confidence': confidences[i]
                })
        
        return detections
    
    def detect(self, frame, conf_threshold=0.3, nms_threshold=0.5):
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            conf_threshold: Confidence threshold
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            list: Detected objects with bounding boxes
        """
        blob, ratio, orig_shape = self.preprocess_image(frame)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
        detections = self.process_output(
            outputs, 
            conf_threshold=conf_threshold, 
            nms_threshold=nms_threshold, 
            img_shape=orig_shape, 
            ratio=ratio
        )
        
        return detections
