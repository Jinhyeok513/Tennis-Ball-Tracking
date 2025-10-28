import argparse, os, cv2, numpy as np, pandas as pd


def clip_box(x, y, w, h, W, H):
    """
    Clamp a box (x, y, w, h) to image bounds [0..W-1] x [0..H-1].
    Inputs may be floats; outputs are integer pixel indices.
    Ensures width/height are non-negative and the box stays inside the frame.
    """
    x2, y2 = x + w, y + h  # (unused, kept for clarity)
    # Clamp top-left corner, then clamp size so (x+w) <= W and (y+h) <= H
    x = max(0, min(W - 1, int(round(x))))
    y = max(0, min(H - 1, int(round(y))))
    w = max(0, min(W - x, int(round(w))))
    h = max(0, min(H - y, int(round(h))))
    return x, y, w, h


def draw_box(img, x, y, w, h, color=(0, 255, 255)):
    """Draw a rectangle with a fixed thickness around (x, y, w, h)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def main():
    # ------------------------------
    # CLI: video path, predictions CSV, output CSV
    # ------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Can't open video")

    # Basic video meta
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280

    # ------------------------------
    # Load per-frame proposal boxes from args.preds
    # Expected columns: frame,x,y,width,height  (center/xy? -> here interpreted as top-left xy)
    # ------------------------------
    preds = pd.read_csv(args.preds)
    preds.columns = [c.strip() for c in preds.columns]
    assert all(c in preds.columns for c in ["frame", "x", "y", "width", "height"])
    pred_map = {
        int(r.frame): (float(r.x), float(r.y), float(r.width), float(r.height))
        for _, r in preds.iterrows()
    }

    # ------------------------------
    # Resume previous labeling if output CSV exists
    # done: dict frame -> (x, y, w, h) or None (explicit missing)
    # ------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        gt = pd.read_csv(args.out)
        gt.columns = [c.strip() for c in gt.columns]
        for _, r in gt.iterrows():
            f = int(r.frame)
            vals = [r.x, r.y, r.width, r.height]
            # Vectorized NaN check: if any NaN in a row, treat as "missing"
            if pd.isna(vals).any():
                done[f] = None
            else:
                done[f] = (float(r.x), float(r.y), float(r.width), float(r.height))

    # Interactive window
    win = "quick_gt"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Iterate frames sequentially
    f = 0
    while f < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break

        # Proposed box for this frame: prediction if available, else a small center box
        prop = pred_map.get(f, None)
        if prop is None:
            prop = (W // 2 - 7, H // 2 - 7, 14, 14)

        x, y, w, h = clip_box(*prop, W, H)

        # If this frame already labeled, prefer that
        if f in done and done[f] is not None:
            x, y, w, h = clip_box(*done[f], W, H)

        # ------------------------------
        # Per-frame edit/accept loop
        # ------------------------------
        while True:
            vis = frame.copy()
            draw_box(vis, x, y, w, h, (0, 255, 255))
            cv2.putText(
                vis,
                "frame {} | a=accept  m=missing  arrows=move  +/-=resize  s=save  q=quit".format(
                    f
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)

            # waitKey returns special codes for arrow keys:
            # left=81, up=82, right=83, down=84 (on many OpenCV builds)
            k = cv2.waitKey(0) & 0xFF

            if k == ord("a"):
                # Accept current box for this frame
                done[f] = (x, y, w, h)
                break
            elif k == ord("m"):
                # Mark as missing (no box for this frame)
                done[f] = None
                break
            elif k in (81, 82, 83, 84):
                # Nudge box position by 1px per keypress
                if k == 81:  # left
                    x -= 1
                elif k == 83:  # right
                    x += 1
                elif k == 82:  # up
                    y -= 1
                elif k == 84:  # down
                    y += 1
            elif k in (ord("+"), ord("=")):
                # Uniformly enlarge box
                w += 1
                h += 1
            elif k in (ord("-"), ord("_")):
                # Uniformly shrink box, keep a minimum size
                w = max(2, w - 1)
                h = max(2, h - 1)
            elif k == ord("s"):
                # Save current progress for ALL frames (0..total-1)
                frames = list(range(total))
                xcol, ycol, wcol, hcol = [], [], [], []
                for ff in frames:
                    val = done.get(ff, np.nan)
                    if val is None:
                        # Explicit missing -> NaNs in CSV
                        xcol.append(np.nan)
                        ycol.append(np.nan)
                        wcol.append(np.nan)
                        hcol.append(np.nan)
                    elif isinstance(val, tuple):
                        # Clip before saving to guarantee valid coordinates
                        xx, yy, ww, hh = clip_box(*val, W, H)
                        xcol.append(xx)
                        ycol.append(yy)
                        wcol.append(ww)
                        hcol.append(hh)
                    else:
                        # Unknown/untouched -> NaNs
                        xcol.append(np.nan)
                        ycol.append(np.nan)
                        wcol.append(np.nan)
                        hcol.append(np.nan)
                out_df = pd.DataFrame(
                    {
                        "frame": frames,
                        "x": xcol,
                        "y": ycol,
                        "width": wcol,
                        "height": hcol,
                    }
                )
                out_df.to_csv(args.out, index=False)
                print("[saved progress]", args.out)
            elif k == ord("q"):
                # Save and exit the tool immediately
                frames = list(range(total))
                xcol, ycol, wcol, hcol = [], [], [], []
                for ff in frames:
                    val = done.get(ff, np.nan)
                    if val is None:
                        xcol.append(np.nan)
                        ycol.append(np.nan)
                        wcol.append(np.nan)
                        hcol.append(np.nan)
                    elif isinstance(val, tuple):
                        xx, yy, ww, hh = clip_box(*val, W, H)
                        xcol.append(xx)
                        ycol.append(yy)
                        wcol.append(ww)
                        hcol.append(hh)
                    else:
                        xcol.append(np.nan)
                        ycol.append(np.nan)
                        wcol.append(np.nan)
                        hcol.append(np.nan)
                out_df = pd.DataFrame(
                    {
                        "frame": frames,
                        "x": xcol,
                        "y": ycol,
                        "width": wcol,
                        "height": hcol,
                    }
                )
                out_df.to_csv(args.out, index=False)
                print("[OK] Saved:", args.out)
                cv2.destroyAllWindows()
                return

        # Move to next frame once accepted/missing is chosen
        f += 1

    # ------------------------------
    # Final save after the last frame
    # ------------------------------
    frames = list(range(total))
    xcol, ycol, wcol, hcol = [], [], [], []
    for ff in frames:
        val = done.get(ff, np.nan)
        if val is None:
            xcol.append(np.nan)
            ycol.append(np.nan)
            wcol.append(np.nan)
            hcol.append(np.nan)
        elif isinstance(val, tuple):
            xx, yy, ww, hh = clip_box(*val, W, H)
            xcol.append(xx)
            ycol.append(yy)
            wcol.append(ww)
            hcol.append(hh)
        else:
            xcol.append(np.nan)
            ycol.append(np.nan)
            wcol.append(np.nan)
            hcol.append(np.nan)

    out_df = pd.DataFrame(
        {"frame": frames, "x": xcol, "y": ycol, "width": wcol, "height": hcol}
    )
    out_df.to_csv(args.out, index=False)
    print("[OK] Saved:", args.out)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
