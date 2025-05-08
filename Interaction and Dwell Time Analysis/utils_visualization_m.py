"""
Visualization utilities for video analytics.
"""
import cv2
import numpy as np

class VideoVisualizer:
    """Visualization utilities for video analytics."""
    
    def __init__(self, colors):
        """
        Initialize the visualizer.
        
        Args:
            colors: Dictionary of color definitions (BGR format)
        """
        self.colors = colors
    
    def draw_tracks(self, frame, tracks, dwell_times):
        """
        Draw tracked objects with IDs and dwell times.
        
        Args:
            frame: Video frame
            tracks: Dictionary of {track_id: box}
            dwell_times: Dictionary of {track_id: dwell_time_seconds}
            
        Returns:
            ndarray: Frame with visualizations
        """
        for track_id, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['track_box'], 2)
            
            # Display ID and dwell time
            dwell_seconds = int(dwell_times.get(track_id, 0)[1])       # ******* remove [1] if not looking for modified
            label = f'ID {track_id} | Dwell time ({dwell_seconds}s)'
            
            # Draw label background
            cv2.rectangle(frame, (x1, y2 - 30), (x1 + 10 + len(label) * 15, y2), 
                          self.colors['label_bg'], -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y2 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['label_text'], 2)
            
        return frame
    
    def draw_rois(self, frame, rois):
        """
        Draw ROIs on the frame.
        
        Args:
            frame: Video frame
            rois: List of ROI polygons
            
        Returns:
            ndarray: Frame with ROIs drawn
        """
        for roi in rois:
            pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=self.colors['roi'], thickness=2)
            
        return frame
    
    def draw_alert(self, frame, message, position=(50, 50)):
        """
        Draw an alert message on the frame.
        
        Args:
            frame: Video frame
            message: Alert message text
            position: (x, y) position for the text
            
        Returns:
            ndarray: Frame with alert drawn
        """
        cv2.putText(frame, message, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['alert'], 3)
        return frame
    
    def draw_interactions(self, frame, interactions, fps):
        """
        Draw interaction lines and durations.
        
        Args:
            frame: Video frame
            interactions: List of interaction dictionaries
            fps: Frames per second
            
        Returns:
            ndarray: Frame with interactions drawn
        """
        for interaction in interactions:
            ids = interaction['ids']
            points = interaction['points']
            total_frames = interaction['total_frames']
            
            duration_s = total_frames / fps
            
            # Draw arrow connecting the points
            p1, p2 = points
            cv2.arrowedLine(frame, p1, p2, self.colors['interaction'], 2, tipLength=0.2)
            
            # Draw ID label at starting point
            id1, id2 = ids
            cv2.putText(frame, f"{id1}->{id2}", 
                       (p1[0], p1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['interaction'], 2)
            
            # Draw duration at midpoint
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            
            # Draw duration background
            cv2.rectangle(frame, (mx, my - 30), (mx + 100, my), 
                         self.colors['label_bg'], -1)
            
            # Draw duration text
            cv2.putText(frame, f"{duration_s:.1f}s", 
                       (mx, my - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                       self.colors['label_text'], 2)
            
        return frame
