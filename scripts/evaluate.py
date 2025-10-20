#!/usr/bin/env python3
"""
Script to calculate metrics against ground truth.
Usage: python evaluate.py --predictions results/classical/predictions.csv --ground_truth data/annotations/test_match_gt.csv
"""

import argparse
import pandas as pd
import numpy as np

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def calculate_cle(pred, gt):
    """Calculate Center Location Error"""
    pred_center = [(pred[0] + pred[2])/2, (pred[1] + pred[3])/2]
    gt_center = [(gt[0] + gt[2])/2, (gt[1] + gt[3])/2]
    return np.sqrt((pred_center[0] - gt_center[0])**2 + (pred_center[1] - gt_center[1])**2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--ground_truth', required=True)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    gt_df = pd.read_csv(args.ground_truth)

    ious = []
    cles = []

    for _, row in pred_df.iterrows():
        frame = row['frame']
        gt_row = gt_df[gt_df['frame'] == frame]
        if not gt_row.empty:
            pred_box = [row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']]
            gt_box = [gt_row['x'].values[0], gt_row['y'].values[0],
                     gt_row['x'].values[0] + gt_row['width'].values[0],
                     gt_row['y'].values[0] + gt_row['height'].values[0]]

            iou = calculate_iou(pred_box, gt_box)
            cle = calculate_cle(pred_box, gt_box)
            ious.append(iou)
            cles.append(cle)

    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"Mean CLE: {np.mean(cles):.4f}")

if __name__ == '__main__':
    main()