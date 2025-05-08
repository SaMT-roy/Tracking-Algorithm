"""
Region of Interest (ROI) analysis utilities.
"""
import cv2
import numpy as np

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.
    
    Args:
        point: (x, y) coordinates of the point
        polygon: List of (x, y) coordinates forming the polygon
        
    Returns:
        bool: True if the point is inside the polygon, False otherwise
    """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def roi_diagonal(roi):
    """
    Calculate the diagonal length of an ROI.
    
    Args:
        roi: List of (x, y) coordinates forming the polygon
        
    Returns:
        float: Diagonal length of the ROI
    """
    xs, ys = zip(*roi)
    return np.hypot(max(xs) - min(xs), max(ys) - min(ys))

class ROIAnalyzer:
    """Region of Interest analyzer for tracking objects in defined areas."""
    
    def __init__(self, rois):
        """
        Initialize the ROI analyzer.
        
        Args:
            rois: List of polygons defining the ROIs
        """
        self.rois = rois
        self.roi_diagonals = [roi_diagonal(roi) for roi in rois]
        
    def filter_detections_by_roi(self, detections):
        """
        Filter detections to only include those inside any ROI.
        
        Args:
            detections: List of detection dictionaries with 'box' key
            
        Returns:
            list: Filtered bounding boxes [x1, y1, x2, y2]
        """
        roi_filtered_boxes = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centroid
            
            # Check if centroid falls inside any ROI
            for roi in self.rois:
                if point_in_polygon((cx, cy), roi):
                    roi_filtered_boxes.append(det['box'])
                    break  # No need to check other ROIs
                    
        return roi_filtered_boxes
    
    def check_occupancy(self, tracks, roi_index=0):
        """
        Check if a specific ROI is occupied by any tracked object.
        
        Args:
            tracks: Dictionary of {track_id: box}
            roi_index: Index of the ROI to check
            
        Returns:
            bool: True if the ROI is occupied, False otherwise
        """
        if roi_index >= len(self.rois):
            return False
            
        roi = self.rois[roi_index]
        
        for _, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centroid
            
            if point_in_polygon((cx, cy), roi):
                return True
                
        return False
    
    def get_objects_in_rois(self, tracks):
        """
        Get objects in each ROI.
        
        Args:
            tracks: Dictionary of {track_id: box}
            
        Returns:
            list: Lists of (track_ids, centroids) for each ROI
        """
        roi_objects = []
        
        for roi in self.rois:
            roi_ids, roi_cents = [], []
            
            for tid, box in tracks.items():
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if point_in_polygon((cx, cy), roi):
                    roi_ids.append(tid)
                    roi_cents.append((cx, cy))
            
            roi_objects.append((roi_ids, roi_cents))
            
        return roi_objects
