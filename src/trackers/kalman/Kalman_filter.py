# Kalman-based tennis-ball tracker with classic CV cues (color, motion, blobness)

import os, csv, argparse
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

# ---------------- Params ----------------
# HSV bounds for yellow (tennis ball) and whiteness thresholds for highlights
# Motion quantiles to normalize flow; net band weight fraction for suppressing net region
YELLOW_H = (22, 75)
YELLOW_SV = (70, 90)
WHITE_SMAX = 95
WHITE_VMIN = 205
MOTION_LOW_Q = 0.75
MOTION_HIGH_Q = 0.995
NET_BAND_FRAC = 0.09

# expected pixel diameter vs image height (tuned for 720p courts)
# The ball appears slightly larger near the bottom of the frame due to perspective.
D_TOP, D_BOT = 9.5, 14.5


# ---------------- Utils ----------------
# Expected ball diameter at a given y (cy), linearly interpolated by image height
def expected_diam_for_y(cy, H, d_top=D_TOP, d_bot=D_BOT):
    a = float(np.clip(cy / max(1.0, float(H)), 0, 1))
    return float(d_top + (d_bot - d_top) * a)


# Safe clamp helper
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# Robustly open a VideoWriter by trying multiple codecs and extensions
def open_writer_safely(base_path, fps, size):
    trials = [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]
    for fourcc_str, ext in trials:
        p = os.path.splitext(base_path)[0] + ext
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if vw.isOpened():
            print(f"[OK] VideoWriter {fourcc_str} -> {p}")
            return vw, p
    raise RuntimeError("VideoWriter open failed")


# Build a soft weight mask penalizing the net band across the middle of the frame
def build_net_weight(H, W, net_w=0.35):
    y0 = int(0.50 * H)
    band = int(NET_BAND_FRAC * H)
    w = np.ones((H, W), np.float32)
    y1, y2 = max(0, y0 - band // 2), min(H, y0 + band // 2)
    w[y1:y2, :] = float(np.clip(net_w, 0, 1))
    return w


# Build a soft court mask from the first frame: blues/cyans are likely court pixels
# Keeps a minimum floor so the mask never fully zeroes out (softer gating)
def build_court_soft(bgr, floor=0.35):
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, Sh, Vh = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    court = ((Hh >= 110) & (Hh <= 165) & (Sh >= 40) & (Vh >= 40)).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    court = cv2.morphologyEx(court, cv2.MORPH_CLOSE, k, 2)
    top = int(0.12 * H)
    court[:top, :] = 0
    return floor + (1.0 - floor) * court


# Hard yellow weight based on HSV thresholds; lightly blurred to reduce noise
def yellow_weight(hsv):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = (
        (H >= YELLOW_H[0])
        & (H <= YELLOW_H[1])
        & (S >= YELLOW_SV[0])
        & (V >= YELLOW_SV[1])
    ).astype(np.float32)
    if w.max() > 0:
        w = cv2.GaussianBlur(w, (3, 3), 0)
    return w


# Whiteness score (bright + low saturation) to capture specular ball highlights
# Later, this is gated by motion to avoid locking onto static white lines.
def whiteness(hsv):
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    s_w = np.clip((WHITE_SMAX - S) / max(1.0, WHITE_SMAX), 0, 1)
    v_w = np.clip((V - WHITE_VMIN) / max(1.0, 255 - WHITE_VMIN), 0, 1)
    return (0.2 + 0.8 * (0.65 * v_w + 0.35 * s_w)).astype(np.float32)


# Difference-of-Gaussians “blobness” to emphasize small, round-ish structures
def dog_blobness(gray):
    g1 = cv2.GaussianBlur(gray, (0, 0), 0.8)
    g2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    dog = cv2.absdiff(g1, g2).astype(np.float32)
    mmin, mmax = float(dog.min()), float(dog.max())
    return (dog - mmin) / (mmax - mmin + 1e-6)


# ---------------- Kalman ----------------
# Constant-velocity Kalman for (x, y, vx, vy). Includes simple gate & miss logic.
class KBall:
    def __init__(self, fps, sigma_a=360.0, r_px=4.0, gate_px=110.0):
        dt = 1.0 / max(1.0, float(fps))
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        # Process noise from constant-acceleration model projected into CV state
        qx = (dt**4) * (sigma_a**2) / 4.0
        qv = (dt**2) * (sigma_a**2)
        self.kf.Q = np.diag([qx, qx, qv, qv])
        # Measurement noise (pixel-level)
        self.kf.R = np.diag([r_px**2, r_px**2])
        # Initial covariance
        self.kf.P = np.eye(4) * 10.0
        # Gating and lifecycle
        self.base_gate = float(gate_px)
        self.gate = float(gate_px)
        self.initialized = False
        self.miss = 0

    # Predict only if initialized
    def predict(self):
        if not self.initialized:
            return None
        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])

    # Update with measurement z=(x,y). If z is None, apply miss logic and expand gate.
    def update(self, z):
        if z is None:
            self.miss += 1
            self.gate = min(400.0, self.gate * 1.2)
            if self.miss > 8:
                self.initialized = False
                self.miss = 0
                self.gate = self.base_gate
            return
        self.kf.update(np.array([z[0], z[1]], float))
        # Snap position to measurement to avoid drift when the observation is strong
        self.kf.x[0], self.kf.x[1] = z[0], z[1]
        self.miss = 0
        self.gate = max(self.base_gate, self.gate * 0.95)


# ---------------- Main ----------------
def main():
    # CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", default="runs/hardlock")
    ap.add_argument("--court_floor", type=float, default=0.35)
    ap.add_argument("--net_weight", type=float, default=0.35)
    ap.add_argument(
        "--dump_weights",
        action="store_true",
        help="Save court/net/yellow/white/motion/dog/fused of the first processed frame to out_dir/debug_weights",
    )
    args = ap.parse_args()

    # Prepare outputs and video IO
    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Try multiple encoders for robust writing
    vw, vis_path = open_writer_safely(
        os.path.join(args.out_dir, "vis.mp4"), fps, (W, H)
    )
    preds_csv = os.path.join(args.out_dir, "preds.csv")
    fcsv = open(preds_csv, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=["frame", "x", "y", "width", "height"])
    writer.writeheader()

    # Build soft court/net masks from the first frame (static through the run)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    assert ok
    court_soft = build_court_soft(first, args.court_floor)
    net_soft = build_net_weight(H, W, args.net_weight)
    print(f"[MASK] mode=SOFT_ONLY court_floor={args.court_floor}")
    print(f"[NET ] net_weight={args.net_weight}, band_frac={NET_BAND_FRAC}")

    # Optionally dump the initial masks for debugging/inspection
    if args.dump_weights:
        dbg_dir = os.path.join(args.out_dir, "debug_weights")
        os.makedirs(dbg_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(dbg_dir, "court_soft.png"),
            (np.clip(court_soft, 0, 1) * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(dbg_dir, "net_soft.png"),
            (np.clip(net_soft, 0, 1) * 255).astype(np.uint8),
        )
        print("[DUMP] saved court_soft.png / net_soft.png ->", dbg_dir)

    # Initial position scan
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    init_xy = None
    for _ in range(20):
        ok, f = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Farnebäck optical flow for motion magnitude
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 13, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # Normalize motion by robust quantiles (suppresses weak background motion)
            lo = np.quantile(mag, MOTION_LOW_Q)
            hi = np.quantile(mag, MOTION_HIGH_Q)
            m = np.clip((mag - lo) / (hi - lo + 1e-6), 0, 1)
            # Yellow cue
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            yel = yellow_weight(hsv)
            # Fuse early cues with court & net priors
            sc = m * yel * court_soft * net_soft
            # Pick best location if score is good enough
            _, mx, _, loc = cv2.minMaxLoc(sc.astype(np.float32))
            if mx > 0.25:
                init_xy = (float(loc[0]), float(loc[1]))
                break
        prev_gray = gray
    # Fallback: center-ish init if nothing confident found
    if init_xy is None:
        init_xy = (W * 0.5, H * 0.45)

    k = KBall(fps=fps)
    k.kf.x = np.array([init_xy[0], init_xy[1], 0.0, 0.0], float)
    k.initialized = True

    # -------- Main loop: per-frame detection → gating → Kalman update → visualization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_i = 0
    prev_gray = None
    last_good = None
    saved_once = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pred = k.predict()
        vx = float(k.kf.x[2]) if k.initialized else 0.0
        vy = float(k.kf.x[3]) if k.initialized else 0.0

        # Compute motion map when possible
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 13, 3, 5, 1.2, 0
            )
            fx, fy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(fx**2 + fy**2)
            lo = np.quantile(mag, MOTION_LOW_Q)
            hi = np.quantile(mag, MOTION_HIGH_Q)
            mot = np.clip((mag - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        else:
            fx = fy = mot = np.zeros_like(gray, np.float32)

        # Color cues: yellow dominant; white only if it is moving (motion gate)
        yel = yellow_weight(hsv)
        white = whiteness(hsv)
        q70 = np.quantile(mot, MOTION_LOW_Q) if prev_gray is not None else 1e9
        white_eff = np.where(mot > q70, white * mot, 0.0).astype(np.float32)
        color = np.maximum(yel, white_eff)

        # Blobness cue (DoG)
        dog = dog_blobness(gray)

        # Fuse cues with multiplicative combination and court/net soft priors
        fused = (mot**1.0) * (0.2 + 0.8 * color) * (0.5 + 0.5 * dog)
        fused *= court_soft
        fused *= net_soft

        # Robust normalization by mid-high quantiles to highlight peaks
        ql = np.quantile(fused, 0.6) if fused.size > 0 else 0.0
        qh = np.quantile(fused, 0.995) if fused.size > 0 else 1.0
        fused = np.clip((fused - ql) / max(1e-6, (qh - ql)), 0, 1)

        # Optionally dump per-cue maps once for debugging
        if args.dump_weights and not saved_once:
            dbg_dir = os.path.join(args.out_dir, "debug_weights")
            os.makedirs(dbg_dir, exist_ok=True)

            def save01(name, arr01):
                cv2.imwrite(
                    os.path.join(dbg_dir, name),
                    (np.clip(arr01, 0, 1) * 255).astype(np.uint8),
                )

            save01("yellow_weight.png", yel)
            save01("white_weight.png", white)
            save01("motion_norm.png", mot)
            save01("dog_blob.png", dog)
            save01("fused.png", np.clip(fused, 0, 1))
            print("[DUMP] saved yellow/white/motion/dog/fused ->", dbg_dir)
            saved_once = True

        # Peak picking on the fused map via dilated local maxima
        thr = 0.5 * float(np.max(fused)) if fused.size > 0 else 1.0
        r = 5
        dil = cv2.dilate(
            fused, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        )
        maxima = (fused >= dil - 1e-12) & (fused >= thr)
        ys, xs = np.where(maxima)

        # Evaluate candidates with geometric heuristics (size/circularity) + motion alignment
        best, best_sc = None, -1.0
        for yy, xx in zip(ys, xs):
            # Expected ball diameter at this y (perspective-aware)
            d_exp = expected_diam_for_y(yy, H)
            wmin, wmax = d_exp * 0.65, d_exp * 1.35
            x1 = int(clamp(xx - wmax / 2, 0, W - 1))
            y1 = int(clamp(yy - wmax / 2, 0, H - 1))
            x2 = int(clamp(xx + wmax / 2, 1, W))
            y2 = int(clamp(yy + wmax / 2, 1, H))
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Circularity check using Otsu binarization on the local patch
            grayp = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, bp = cv2.threshold(grayp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(bp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circ = 0.0
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                peri = cv2.arcLength(c, True)
                area = cv2.contourArea(c)
                if peri > 0 and area > 2:
                    circ = float(4.0 * np.pi * (area / (peri * peri)))
            if circ < 0.45:
                continue

            # Reject if candidate box is far from expected size
            rr = max(y2 - y1, x2 - x1)
            if rr < wmin or rr > wmax * 1.2:
                continue

            # If we have a velocity estimate, prefer candidates whose local flow matches it
            if pred is not None and (abs(vx) + abs(vy)) > 0.8:
                ux = float(fx[yy, xx])
                uy = float(fy[yy, xx])
                un = np.hypot(ux, uy) + 1e-6
                cos = (ux * vx + uy * vy) / (un * np.hypot(vx, vy))
                if cos < 0.2:
                    continue

            # Score by fused-map strength
            sc = float(fused[yy, xx])
            if sc > best_sc:
                best_sc = sc
                best = (xx, yy, float(rr))

        # Final selection with temporal/gate logic to avoid snapping to players/lines
        chosen = None
        if best is not None:
            if last_good is None:
                chosen = (best[0], best[1])
            else:
                dx = best[0] - last_good[0]
                dy = best[1] - last_good[1]
                if (
                    dx * dx + dy * dy <= (expected_diam_for_y(best[1], H) ** 2) * 2.0
                    or best_sc > 0.7
                ):
                    chosen = (best[0], best[1])

        # Kalman update (None → miss logic; otherwise measurement update)
        if chosen is not None:
            k.update(chosen)
            last_good = (chosen[0], chosen[1])
        else:
            k.update(None)

        # ------------- Visualization & logging
        vis = frame.copy()
        if k.initialized:
            cx, cy = float(k.kf.x[0]), float(k.kf.x[1])
            d_draw = int(max(10, expected_diam_for_y(cy, H)))
            x1, y1 = int(cx - d_draw / 2), int(cy - d_draw / 2)
            x2, y2 = x1 + d_draw, y1 + d_draw
            # Yellow tracking box and a green center dot
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)
            # Indicate prediction-only frames
            if chosen is None:
                cv2.putText(
                    vis,
                    "PRED ONLY",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            # Log detection box to CSV as (x,y,width,height)
            writer.writerow(
                {"frame": frame_i, "x": x1, "y": y1, "width": d_draw, "height": d_draw}
            )
        else:
            # Not initialized (or lost target for too long) → searching state
            cv2.putText(
                vis,
                "SEARCHING...",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2,
            )

        # Write visualization frame and step forward
        vw.write(vis)
        prev_gray = gray
        frame_i += 1

    # Clean up I/O resources
    cap.release()
    vw.release()
    fcsv.close()
    print("[OK] Saved:", vis_path, " , ", preds_csv)


if __name__ == "__main__":
    main()
