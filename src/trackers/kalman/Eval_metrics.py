import argparse, os, math
import pandas as pd
import numpy as np


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_boxes_csv(path, suffix):
    """
    Load a CSV that stores per-frame bounding boxes and normalize its schema.

    1. Accepted input schemas

    1) Top-left + size (XYWH):
       columns = ["frame", "x", "y", "width", "height"]
       Units are pixels. (x, y) is the top-left corner.

    2) Center + radius:
       columns = ["frame", "cx", "cy", "r"]
       Units are pixels. A square box is inferred with width = height = 2*r.
       Converted to XYWH with:
         x = cx - r, y = cy - r, w = 2r, h = 2r

    2. Output schema

    DataFrame with columns:
      ["frame", f"x{suffix}", f"y{suffix}", f"w{suffix}", f"h{suffix}"]

    Notes
    -----
    - The function is tolerant to accidental whitespace in column names.
    - All numeric columns are cast to float (frame to int).
    - Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(path)
    if "frame" not in df.columns:
        raise ValueError(f"{path}: missing 'frame' column")

    # Normalize column names to avoid issues like " width " vs "width"
    df.columns = [c.strip() for c in df.columns]

    has_xywh = all(c in df.columns for c in ["x", "y", "width", "height"])
    has_center = all(c in df.columns for c in ["cx", "cy", "r"])
    if not (has_xywh or has_center):
        raise ValueError(f"{path}: expected columns (x,y,width,height) or (cx,cy,r)")

    out = pd.DataFrame({"frame": df["frame"].astype(int)})

    if has_xywh:
        # Pass through x, y, width, height as floats
        out[f"x{suffix}"] = df["x"].astype(float)
        out[f"y{suffix}"] = df["y"].astype(float)
        out[f"w{suffix}"] = df["width"].astype(float)
        out[f"h{suffix}"] = df["height"].astype(float)
    else:
        # Convert (cx, cy, r) → (x, y, w, h) assuming a square box
        cx = df["cx"].astype(float)
        cy = df["cy"].astype(float)
        r = df["r"].astype(float)
        w = 2.0 * r
        h = 2.0 * r
        x = cx - r
        y = cy - r
        out[f"x{suffix}"] = x
        out[f"y{suffix}"] = y
        out[f"w{suffix}"] = w
        out[f"h{suffix}"] = h

    return out


def clip_box(x, y, w, h, W, H):
    """
    Clip a box to the image bounds [0, W] * [0, H].

    Any NaN among (x, y, w, h) results in a fully NaN box.

    x, y, w, h : float
        Box in XYWH (top-left origin).
    W, H : int
        Frame width and height (pixels).

    tuple(float, float, float, float)
        Clipped (x, y, w, h). If the original box is completely outside,
        width/height can become 0.0.
    """
    if any(pd.isna([x, y, w, h])):
        return np.nan, np.nan, np.nan, np.nan

    # Bottom/right corner
    x2 = x + w
    y2 = y + h

    # Clamp coordinates to image range
    x1c = max(0.0, min(float(W), x))
    y1c = max(0.0, min(float(H), y))
    x2c = max(0.0, min(float(W), x2))
    y2c = max(0.0, min(float(H), y2))

    # Recompute width/height after clipping
    wc = max(0.0, x2c - x1c)
    hc = max(0.0, y2c - y1c)
    return x1c, y1c, wc, hc


def iou_xywh(ax, ay, aw, ah, bx, by, bw, bh, W, H):
    """
    Compute IoU between two XYWH boxes after clipping to the frame.

    Returns
    -------
    float
        Intersection-over-Union in [0,1]. NaN if either box is invalid
        after clipping (e.g., empty area or NaN input).
    """
    # Clip both boxes to avoid negative areas or out-of-bounds intersections
    ax, ay, aw, ah = clip_box(ax, ay, aw, ah, W, H)
    bx, by, bw, bh = clip_box(bx, by, bw, bh, W, H)

    if any(pd.isna([ax, ay, aw, ah, bx, by, bw, bh])):
        return np.nan

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection rectangle
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    ua = aw * ah + bw * bh - inter  # union area
    if ua <= 0:
        return np.nan

    return inter / ua


def cle_xywh(ax, ay, aw, ah, bx, by, bw, bh):
    """
    Compute Center Location Error (CLE) between two XYWH boxes.
    Euclidean distance (pixels) between the two box centers.
    NaN if any input is NaN.
    """
    if any(pd.isna([ax, ay, aw, ah, bx, by, bw, bh])):
        return np.nan

    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return float(math.hypot(acx - bcx, acy - bcy))


def main():
    """
    CLI entry point.

    Arguments
    ---------
    --preds : path to predictions CSV
    --gt : path to ground-truth CSV
    --frame_width, --frame_height : frame dimensions (pixels)
    --out_dir : directory to save outputs (CSV + text summary)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds.csv path")
    ap.add_argument("--gt", required=True, help="gt.csv path")
    ap.add_argument("--frame_width", type=int, required=True)
    ap.add_argument("--frame_height", type=int, required=True)
    ap.add_argument("--out_dir", default="metrics")
    args = ap.parse_args()

    W, H = int(args.frame_width), int(args.frame_height)
    ensure_dir(args.out_dir)

    # Load and tag columns with suffixes to avoid collisions after merge
    df_pred = load_boxes_csv(args.preds, "_pred")
    df_gt = load_boxes_csv(args.gt, "_gt")

    # Outer merge keeps frames that appear in either file
    df = (
        pd.merge(df_gt, df_pred, on="frame", how="outer")
        .sort_values("frame")
        .reset_index(drop=True)
    )

    # Compute per-frame IoU & CLE
    ious = []
    cles = []
    for _, r in df.iterrows():
        ax, ay, aw, ah = (
            r.get("x_pred"),
            r.get("y_pred"),
            r.get("w_pred"),
            r.get("h_pred"),
        )
        bx, by, bw, bh = r.get("x_gt"), r.get("y_gt"), r.get("w_gt"), r.get("h_gt")
        iou = iou_xywh(ax, ay, aw, ah, bx, by, bw, bh, W, H)
        cle = cle_xywh(ax, ay, aw, ah, bx, by, bw, bh)
        ious.append(iou)
        cles.append(cle)

    df["IoU"] = ious
    df["CLE"] = cles

    # Helper: mean that ignores NaNs and returns float('nan') if empty
    def valid_mean(series):
        return float(series.dropna().mean()) if series.notna().any() else float("nan")

    # Aggregate IoU statistics
    mean_iou = valid_mean(df["IoU"])
    succ_05 = (
        float((df["IoU"] >= 0.5).mean()) if df["IoU"].notna().any() else float("nan")
    )
    succ_03 = (
        float((df["IoU"] >= 0.3).mean()) if df["IoU"].notna().any() else float("nan")
    )

    # Aggregate CLE statistics (lower is better)
    mean_cle = valid_mean(df["CLE"])
    prec_10 = (
        float((df["CLE"] <= 10).mean()) if df["CLE"].notna().any() else float("nan")
    )
    prec_20 = (
        float((df["CLE"] <= 20).mean()) if df["CLE"].notna().any() else float("nan")
    )

    # Basic accounting for how many frames contributed to each metric
    n_frames = int(df.shape[0])
    n_valid_iou = int(df["IoU"].notna().sum())
    n_valid_cle = int(df["CLE"].notna().sum())

    # Save per-frame table (one row per frame id)
    per_frame_path = os.path.join(args.out_dir, "metrics_kf.csv")
    df.to_csv(per_frame_path, index=False)

    # Save a compact text summary for quick inspection or logging
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== IoU / CLE Summary ===\n")
        f.write(f"Frames (merged): {n_frames}\n")
        f.write(f"Valid IoU frames: {n_valid_iou}\n")
        f.write(f"Valid CLE frames: {n_valid_cle}\n")

        f.write("\n-- IoU --\n")
        f.write(
            f"Mean IoU: {mean_iou:.4f}\n"
            if not math.isnan(mean_iou)
            else "Mean IoU: NaN\n"
        )
        f.write(
            f"Success@IoU>=0.5: {succ_05*100:.1f}%\n"
            if not math.isnan(succ_05)
            else "Success@IoU>=0.5: NaN\n"
        )
        f.write(
            f"Success@IoU>=0.3: {succ_03*100:.1f}%\n"
            if not math.isnan(succ_03)
            else "Success@IoU>=0.3: NaN\n"
        )

        f.write("\n-- CLE (px) --\n")
        f.write(
            f"Mean CLE (px): {mean_cle:.2f}\n"
            if not math.isnan(mean_cle)
            else "Mean CLE (px): NaN\n"
        )
        f.write(
            f"Precision@CLE<=10px: {prec_10*100:.1f}%\n"
            if not math.isnan(prec_10)
            else "Precision@CLE<=10px: NaN\n"
        )
        f.write(
            f"Precision@CLE<=20px: {prec_20*100:.1f}%\n"
            if not math.isnan(prec_20)
            else "Precision@CLE<=20px: NaN\n"
        )

    print("[OK] Saved:", per_frame_path)
    print("[OK] Saved:", summary_path)


if __name__ == "__main__":
    main()
