"""
Kalman filter tracker for tennis ball tracking.
"""

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from .base_tracker import BaseTracker

class KalmanTracker(BaseTracker):
    def __init__(self, process_noise=0.1, measurement_noise=0.1):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000
        self.kf.R *= measurement_noise
        self.kf.Q *= process_noise
        self.predictions = []
        self.initialized = False

    def track(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # For simplicity, assume we have detections from classical method
            # In practice, integrate with detection
            if not self.initialized:
                # Initialize with first detection
                self.kf.x = np.array([100, 200, 0, 0])  # x, y, vx, vy
                self.initialized = True

            self.kf.predict()
            # Update with measurement (simulated)
            measurement = np.array([self.kf.x[0], self.kf.x[1]])
            self.kf.update(measurement)

            x, y = self.kf.x[0], self.kf.x[1]
            self.predictions.append({
                'frame': frame_idx,
                'x': x, 'y': y, 'width': 10, 'height': 10,
                'confidence': 0.9
            })

            frame_idx += 1

        cap.release()

    def get_predictions(self):
        return self.predictions