#!/usr/bin/env python3
"""
Main script to run a tracker on a video.
Usage: python run_tracking.py --method classical --video data/raw_videos/test_match.mp4
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run tennis ball tracking')
    parser.add_argument('--method', choices=['classical', 'kalman', 'yolo', 'tracknet'], required=True)
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', help='Path to output video')
    args = parser.parse_args()

    # Import tracker based on method
    if args.method == 'classical':
        from trackers.classical_tracker import ClassicalTracker
        tracker = ClassicalTracker()
    elif args.method == 'kalman':
        from trackers.kalman_tracker import KalmanTracker
        tracker = KalmanTracker()
    elif args.method == 'yolo':
        from trackers.deep_learning_tracker import DeepLearningTracker
        tracker = DeepLearningTracker('yolo')
    elif args.method == 'tracknet':
        from trackers.deep_learning_tracker import DeepLearningTracker
        tracker = DeepLearningTracker('tracknet')

    # Run tracking
    print(f"Running {args.method} tracking on {args.video}")
    # tracker.track(args.video, args.output)

if __name__ == '__main__':
    main()