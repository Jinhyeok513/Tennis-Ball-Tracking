import os
import csv
import cv2
import math
import argparse
import numpy as np
from collections import OrderedDict


def expected_diam_for_y(y, H, d_top, d_bot):
    """Return an expected bounding-box diameter (in px) by linearly
    interpolating between d_top (near the top of the image) and d_bot
    (near the bottom) based on the normalized vertical position y/H.
    This provides a simple perspective prior for ball size."""
    a = float(np.clip(y / max(1.0, float(H)), 0.0, 1.0))
    return float(d_top + (d_bot - d_top) * a)


def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


def load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    return frame if ok else None


def draw_ui(frame, msg_lines, center_xy, box_d):
    """Overlay helper UI onto a frame:
    - optional ball center dot and square box with side length=box_d
    - status text lines in the top-left corner."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw center and box if a click (center) exists for this frame
    if center_xy is not None and box_d is not None:
        cx, cy = center_xy
        d = int(round(box_d))

        # Convert center + diameter to a square ROI; clamp to image bounds
        x1 = clamp_int(cx - d // 2, 0, w - 1)
        y1 = clamp_int(cy - d // 2, 0, h - 1)
        x2 = clamp_int(x1 + d, 1, w)
        y2 = clamp_int(y1 + d, 1, h)

        # Yellow rectangle and a small green center dot
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)

    # Render multi-line HUD text
    y0 = 24
    for line in msg_lines:
        cv2.putText(
            vis,
            line,
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        y0 += 22
    return vis


def interpolate_tracks(clicks, H, d_top, d_bot):
    """
    Turn sparse user clicks into a per-frame box track via linear interpolation.
    - clicks: OrderedDict {frame_idx: (cx, cy)} sorted by frame index
    - For each consecutive pair of clicked frames [f0, f1], linearly interpolate
      the (cx, cy) path and compute an expected diameter per interpolated y.
    - Returns: dict {frame_idx: (x, y, width, height)} with (x, y) being center
      coordinates and width=height=expected diameter (a square box).
    """
    frames = sorted(clicks.keys())
    out = {}
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        (x0, y0), (x1, y1) = clicks[f0], clicks[f1]
        span = f1 - f0
        if span <= 0:
            continue

        # Include both endpoints so the last interpolated point equals the next click
        for k in range(span + 1):
            f = f0 + k
            t = k / float(span)
            x = (1 - t) * x0 + t * x1
            y = (1 - t) * y0 + t * y1

            # Perspective-aware square side length from vertical position
            d = expected_diam_for_y(y, H, d_top, d_bot)
            box_w = box_h = float(d)

            # Store center-format box for this frame
            out[f] = (x, y, box_w, box_h)
    return out


def save_csv(path, mapping):
    """Save a mapping {frame: (cx, cy, w, h)} to CSV with columns:
    frame,x,y,width,height. Creates parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "x", "y", "width", "height"])
        writer.writeheader()
        for fidx in sorted(mapping.keys()):
            x, y, w, h = mapping[fidx]
            writer.writerow(
                {
                    "frame": int(fidx),
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                }
            )


def main():
    # -----------------------------
    # CLI arguments
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--start", type=int, default=0, help="start frame (inclusive)")
    ap.add_argument(
        "--end", type=int, default=-1, help="end frame (inclusive); -1 = to video end"
    )
    ap.add_argument("--stride", type=int, default=10, help="click every K frames")
    ap.add_argument(
        "--d_top", type=float, default=9.5, help="expected diameter near top (px)"
    )
    ap.add_argument(
        "--d_bot", type=float, default=14.5, help="expected diameter near bottom (px)"
    )
    args = ap.parse_args()

    # Open the video and query basic metadata
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Resolve start/end frame indices within [0, total-1]
    start = max(0, args.start)
    end = total - 1 if args.end < 0 else min(args.end, total - 1)
    if end < start:
        raise ValueError("end must be >= start")

    # Build the list of frames at which the user will click (sparse waypoints)
    click_frames = list(range(start, end + 1, max(1, args.stride)))
    # Ensure the last frame is included even if not aligned to the stride
    if click_frames[-1] != end:
        click_frames.append(end)

    clicks = OrderedDict()
    idx = 0

    # State to render a preview box for the current frame
    current_center = None
    current_diam = None

    # Create an interactive window
    win = "sparse_gt_from_clicks"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Mouse callback to record a click at the current target frame
    def on_mouse(event, x, y, flags, userdata):
        nonlocal current_center, current_diam, clicks, idx
        if event == cv2.EVENT_LBUTTONDOWN:
            f = click_frames[idx]
            clicks[f] = (float(x), float(y))
            current_center = (x, y)
            current_diam = expected_diam_for_y(y, H, args.d_top, args.d_bot)

    cv2.setMouseCallback(win, on_mouse)

    # Main interaction loop:
    # - Show the current target frame
    while True:
        f = click_frames[idx]
        frame = load_frame(cap, f)

        if frame is None:
            # Fallback visual if the frame cannot be loaded
            msg = [f"[WARN] cannot read frame {f}", "Press n/b or q"]
            vis = np.zeros((480, 640, 3), np.uint8)
            vis = draw_ui(vis, msg, None, None)
            cv2.imshow(win, vis)
        else:
            # If this frame has a click, show its center and expected box size
            if f in clicks:
                cx, cy = clicks[f]
                current_center = (int(round(cx)), int(round(cy)))
                current_diam = expected_diam_for_y(cy, H, args.d_top, args.d_bot)
            else:
                current_center = None
                current_diam = None

            # Heads-up instructions and status
            msg = [
                f"Video: {os.path.basename(args.video)}",
                f"Frame: {f}  ({idx+1}/{len(click_frames)})",
                f"Clicked: {'yes' if f in clicks else 'no'}   total_clicks={len(clicks)}",
                "Click=mark center, n/Space=next, b/p=prev, u=undo, s=save, q/Esc=save+quit",
            ]
            vis = draw_ui(frame, msg, current_center, current_diam)
            cv2.imshow(win, vis)

        # Keyboard controls (non-blocking with 30 ms delay)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # q or Esc -> exit after final save
            break
        elif key in (ord("n"), 32):  # n or Space -> go to next click target
            idx = min(idx + 1, len(click_frames) - 1)
        elif key in (ord("b"), ord("p")):  # b or p -> go to previous click target
            idx = max(idx - 1, 0)
        elif key == ord("u"):
            # Undo current frame's click (if any)
            fcur = click_frames[idx]
            if fcur in clicks:
                clicks.pop(fcur)
        elif key == ord("s"):
            # Partial save: interpolate using all current clicks, then write CSV
            mapping = {}
            if len(clicks) >= 2:
                mapping = interpolate_tracks(clicks, H, args.d_top, args.d_bot)
            save_csv(args.out, mapping)
            print(f"[SAVE] partial -> {args.out}")

    # Final save on exit (only meaningful if at least 2 clicks exist)
    mapping = {}
    if len(clicks) >= 2:
        mapping = interpolate_tracks(clicks, H, args.d_top, args.d_bot)
    save_csv(args.out, mapping)
    print(f"[OK] saved: {args.out}")
    print(f"[INFO] clicks: {len(clicks)}  interpolated frames: {len(mapping)}")

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
