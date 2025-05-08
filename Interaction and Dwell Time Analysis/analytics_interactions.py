"""
Interaction analysis between objects in different ROIs.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

class InteractionAnalyzer:
    """Analyzer for interactions between objects in different ROIs."""
    
    def __init__(self, roi_diagonals):
        """
        Initialize the interaction analyzer.
        
        Args:
            roi_diagonals: List of diagonal lengths for each ROI
        """
        self.roi_diagonals = roi_diagonals
        self.max_match_distance = 0.5 * sum(roi_diagonals[:2]) 
        
        # Analytics storage
        self.interactions = {}  # (id1, id2) → {'total_frames': int, 'active_start': int}
    
    def analyze_interactions(self, roi_objects, frame_count):
        """
        Analyze interactions between objects in different ROIs.
        
        Args:
            roi_objects: List of (track_ids, centroids) for each ROI
            frame_count: Current frame count
            
        Returns:
            list: Current interaction pairs and their info
        """
        if len(roi_objects) < 2:
            return []
            
        # Extract data for the first two ROIs
        roi1_ids, roi1_cents = roi_objects[0]
        roi2_ids, roi2_cents = roi_objects[1]
        
        # Hungarian matching if both ROIs have objects
        current_keys = set()
        current_interactions = []
        
        if roi1_cents and roi2_cents:
            A = np.array(roi1_cents)  # shape (n1, 2)
            B = np.array(roi2_cents)  # shape (n2, 2)
            
            # Distance matrix
            cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            
            # Apply distance threshold
            cost[cost > self.max_match_distance] = 1e6
            
            # Solve assignment problem
            rows, cols = linear_sum_assignment(cost)
            
            for r, c in zip(rows, cols):
                if cost[r, c] < 1e5:  # Valid match
                    id1, id2 = roi1_ids[r], roi2_ids[c]
                    key = (id1, id2)
                    current_keys.add(key)
                    
                    # Start or resume tracking this interaction
                    rec = self.interactions.get(key)
                    if rec is None:
                        # First time ever seen
                        self.interactions[key] = {
                            'total_frames': 0,
                            'active_start': frame_count
                        }
                        rec = self.interactions[key]
                    elif rec['active_start'] is None:
                        # Resuming after a pause
                        rec['active_start'] = frame_count
                    
                    # Compute cumulative duration (frames)
                    running = (frame_count - rec['active_start']) if rec['active_start'] is not None else 0
                    total_frames = rec['total_frames'] + running
                    
                    # Add to current interactions
                    current_interactions.append({
                        'ids': (id1, id2),
                        'points': (tuple(A[r]), tuple(B[c])),
                        'total_frames': total_frames
                    })
        
        # Update interactions with pause/resume logic
        self._update_interaction_states(current_keys, frame_count)
        
        return current_interactions
    
    def _update_interaction_states(self, current_keys, frame_count):
        """
        Update the state of all interactions.
        
        Args:
            current_keys: Set of currently active interaction keys
            frame_count: Current frame count
        """
        # Resume any interactions that started this frame
        for key in current_keys:
            rec = self.interactions.get(key)
            if rec is None:
                # First time ever seen
                self.interactions[key] = {
                    'total_frames': 0,
                    'active_start': frame_count
                }
            elif rec['active_start'] is None:
                # Resuming after a break
                rec['active_start'] = frame_count
        
        # Pause any interactions that ended this frame
        for key, rec in self.interactions.items():
            if key not in current_keys and rec['active_start'] is not None:
                rec['total_frames'] += (frame_count - rec['active_start'])
                rec['active_start'] = None
    
    def get_interaction_duration(self, interaction_key, frame_count, fps):
        """
        Get the duration of an interaction in seconds.
        
        Args:
            interaction_key: (id1, id2) key
            frame_count: Current frame count
            fps: Frames per second
            
        Returns:
            float: Duration in seconds
        """
        rec = self.interactions.get(interaction_key)
        if rec is None:
            return 0.0
            
        # If still active, add up to last frame
        if rec['active_start'] is not None:
            total = rec['total_frames'] + (frame_count - rec['active_start'])
        else:
            total = rec['total_frames']
            
        return total / fps
    
    def get_all_interaction_durations(self, frame_count, fps, min_duration=0.0):
        """
        Get all interaction durations.
        
        Args:
            frame_count: Current frame count
            fps: Frames per second
            min_duration: Minimum duration to include (seconds)
            
        Returns:
            dict: Dictionary of {(id1, id2): duration_seconds}
        """
        results = {}
        
        for key in self.interactions:
            duration = self.get_interaction_duration(key, frame_count, fps)
            if duration >= min_duration:
                results[key] = duration
                
        return results
