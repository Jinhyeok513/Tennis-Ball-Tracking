"""
Classical tracker using frame differencing and background subtraction.
"""

import cv2
import numpy as np
from .base_tracker import BaseTracker

class ClassicalTracker(BaseTracker):
    def __init__(self, threshold=25, min_area=100, max_area=1000):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.predictions = []

    def track(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    self.predictions.append({
                        'frame': frame_idx,
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'confidence': 0.8
                    })

            prev_gray = gray
            frame_idx += 1

        cap.release()

    def get_predictions(self):
        return self.predictions