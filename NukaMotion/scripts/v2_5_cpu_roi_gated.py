import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

import requests
NUKA_REP_URL = "http://127.0.0.1:8008/rep"



MODEL_PATH = "../models/movenet_lightning_int8.tflite"
CAM_INDEX = 0
IN_SIZE = 192

L_SHOULDER=5; R_SHOULDER=6
L_HIP=11; R_HIP=12
L_KNEE=13; R_KNEE=14
L_ANKLE=15; R_ANKLE=16

SKELETON = [
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, L_KNEE), (R_HIP, R_KNEE),
    (L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE),
]

def letterbox_rgb_u8(frame_bgr, out_size=IN_SIZE):
    h, w = frame_bgr.shape[:2]
    scale = out_size / max(h, w)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_left = (out_size - nw)//2
    pad_top  = (out_size - nh)//2
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    meta = (scale, pad_left, pad_top, w, h)
    return rgb, meta

def kp_to_frame_xy(kp_y, kp_x, meta, in_size=IN_SIZE):
    scale, pad_left, pad_top, fw, fh = meta
    x = (kp_x * in_size - pad_left) / scale
    y = (kp_y * in_size - pad_top) / scale
    x = float(np.clip(x, 0, fw-1))
    y = float(np.clip(y, 0, fh-1))
    return x, y

def angle_3pts(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        if x is None:
            return self.v
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha*x + (1-self.alpha)*self.v
        return float(self.v)

class SquatPractical:
    def __init__(self, down_th=120.0, up_th=155.0, min_score=0.15, cooldown=0.15):
        self.state = "STAND"
        self.count = 0
        self.down_th = down_th
        self.up_th = up_th
        self.min_score = min_score
        self.cooldown = cooldown
        self.last_t = 0.0
        self.ema = EMA(alpha=0.25)
        self.last_valid_ang = None
        self.last_valid_t = 0.0

    def _leg_angle(self, kps, meta, hip_i, knee_i, ankle_i):
        hy,hx,hs = kps[hip_i]
        ky,kx,ks = kps[knee_i]
        ay,ax,as_ = kps[ankle_i]
        if min(hs, ks, as_) < self.min_score:
            return None
        hip = kp_to_frame_xy(hy,hx,meta)
        knee = kp_to_frame_xy(ky,kx,meta)
        ankle = kp_to_frame_xy(ay,ax,meta)
        return angle_3pts(hip, knee, ankle)

    def compute_angle(self, kps, meta):
        la = self._leg_angle(kps, meta, L_HIP, L_KNEE, L_ANKLE)
        ra = self._leg_angle(kps, meta, R_HIP, R_KNEE, R_ANKLE)

        if la is None and ra is None:
            ang, src = None, "no_leg"
        elif la is None:
            ang, src = ra, "R"
        elif ra is None:
            ang, src = la, "L"
        else:
            ang, src = (la+ra)/2.0, "LR"

        if ang is not None and not (30.0 <= ang <= 180.0):
            ang = None

        ang = self.ema.update(ang)

        # hold 时间缩短：避免“站立角度”把你锁死
        now = time.time()
        if ang is None and self.last_valid_ang is not None and (now - self.last_valid_t) < 0.25:
            ang = self.last_valid_ang
            src = src + "+hold"
        elif ang is not None:
            self.last_valid_ang = ang
            self.last_valid_t = now

        return ang, src

    def update(self, ang):
        if ang is None:
            return None
        now = time.time()
        if now - self.last_t < self.cooldown:
            return ang, None, self.count

        event = None
        if self.state == "STAND":
            if ang < self.down_th:
                self.state = "DOWN"
                self.last_t = now
                event = "DOWN"
        else:
            if ang > self.up_th:
                self.state = "STAND"
                self.last_t = now
                self.count += 1
                event = "SQUAT_DONE"
        return ang, event, self.count

def clamp_bbox(x1,y1,x2,y2,w,h):
    x1 = int(max(0, min(w-1, x1)))
    y1 = int(max(0, min(h-1, y1)))
    x2 = int(max(1, min(w,   x2)))
    y2 = int(max(1, min(h,   y2)))
    if x2 <= x1+2 or y2 <= y1+2:
        return None
    return (x1,y1,x2,y2)

def bbox_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def expand_bbox(b, w, h, scale=2.2, min_size=360):
    x1,y1,x2,y2 = b
    cx, cy = bbox_center(b)
    bw = (x2-x1)*scale
    bh = (y2-y1)*scale
    s = max(bw, bh, min_size)
    nx1 = cx - s/2
    ny1 = cy - s/2
    nx2 = cx + s/2
    ny2 = cy + s/2
    return clamp_bbox(nx1, ny1, nx2, ny2, w, h)

def kps_to_bbox(kps, meta, score_th=0.35):
    idxs = [L_SHOULDER,R_SHOULDER,L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE]
    xs, ys = [], []
    for i in idxs:
        y,x,s = kps[i]
        if s >= score_th:
            px, py = kp_to_frame_xy(y,x,meta)
            xs.append(px); ys.append(py)
    # 至少 4 个点才信 bbox
    if len(xs) < 4:
        return None
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

def map_roi_point_to_full(px, py, roi_bbox):
    x1,y1,_,_ = roi_bbox
    return (px + x1, py + y1)

def main():
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter("debug_out.mp4", fourcc, 20.0, (1280, 720))

    det = SquatPractical()

    last_roi = None
    bad_quality_since = None

    # 启动阶段：先强制全图 1 秒
    start_t = time.time()
    force_full_until = start_t + 1.0

    fps_t0, n = time.time(), 0
    print("Running V2.5 (ROI gated + fallback)... Ctrl+C to stop -> debug_out.mp4")

    try:
        while True:
            ok, frame_full = cap.read()
            if not ok:
                continue
            H, W = frame_full.shape[:2]

            now = time.time()
            force_full = now < force_full_until

            use_roi = (last_roi is not None) and (not force_full)
            roi_bbox = last_roi if use_roi else (0,0,W,H)

            x1,y1,x2,y2 = roi_bbox
            frame = frame_full[y1:y2, x1:x2]
            fh, fw = frame.shape[:2]

            rgb, meta = letterbox_rgb_u8(frame, IN_SIZE)
            x = rgb[np.newaxis, ...].astype(np.uint8)

            t0 = time.time()
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()
            kps = interpreter.get_tensor(out["index"])[0,0,:,:]
            infer_ms = (time.time()-t0)*1000.0

            scores = kps[:,2]
            min_s = float(scores.min()); mean_s = float(scores.mean())

            ang, src = det.compute_angle(kps, meta)
            res = det.update(ang)

            # ===== ROI 质量门控 =====
            quality_ok = (mean_s >= 0.35) and (src != "no_leg")
            if not quality_ok:
                if bad_quality_since is None:
                    bad_quality_since = now
            else:
                bad_quality_since = None

            # 质量差持续 0.5s => 回退全图重捕获
            if bad_quality_since is not None and (now - bad_quality_since) > 0.5:
                last_roi = None

            # 只有质量好时才更新 ROI
            if quality_ok:
                b = kps_to_bbox(kps, meta, score_th=0.35)  # ROI coords
                if b is not None:
                    ex = expand_bbox(b, fw, fh, scale=2.2, min_size=360)
                    if ex is not None:
                        rx1,ry1,rx2,ry2 = ex
                        new_roi = clamp_bbox(rx1+x1, ry1+y1, rx2+x1, ry2+y1, W, H)
                        # jump guard
                        if last_roi is not None and new_roi is not None:
                            c0 = np.array(bbox_center(last_roi))
                            c1 = np.array(bbox_center(new_roi))
                            dist = np.linalg.norm(c1 - c0)
                            max_jump = 0.22 * min(W, H)
                            if dist <= max_jump:
                                last_roi = new_roi
                        elif new_roi is not None:
                            last_roi = new_roi

            # ===== overlay text =====
            if res is None:
                text = f"KP/angle low | minS={min_s:.2f} meanS={mean_s:.2f} | {infer_ms:.1f}ms | ROI={'ON' if use_roi else 'OFF'}"
            else:
                ang2, event, cnt = res
                text = f"knee={ang2:.1f} src={src} state={det.state} cnt={cnt} | minS={min_s:.2f} meanS={mean_s:.2f} | {infer_ms:.1f}ms | ROI={'ON' if use_roi else 'OFF'}"
                if event == "DOWN":
                    print("⬇️ DOWN")
                elif event == "SQUAT_DONE":
                    print("✅ SQUAT_DONE | cnt=", cnt)
                    
                    # Send rep to Nuka (short timeout to avoid blocking camera loop)
                    try:
                        requests.post(NUKA_REP_URL, timeout=0.8)
                    except Exception:
                        pass

            # draw ROI rect
            if last_roi is not None:
                a,b,c,d = last_roi
                cv2.rectangle(frame_full, (a,b), (c,d), (0,255,0), 2)

            # draw skeleton on full
            min_draw = 0.15
            draw_pts = {}
            for idx in [L_SHOULDER,R_SHOULDER,L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE]:
                ky,kx,ks = kps[idx]
                if ks >= min_draw:
                    px, py = kp_to_frame_xy(ky,kx,meta)
                    fx, fy = map_roi_point_to_full(px, py, roi_bbox)
                    draw_pts[idx] = (int(fx), int(fy))
                    cv2.circle(frame_full, draw_pts[idx], 4, (0,255,0), -1)
            for a,b in SKELETON:
                if a in draw_pts and b in draw_pts:
                    cv2.line(frame_full, draw_pts[a], draw_pts[b], (0,255,0), 2)

            cv2.putText(frame_full, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,255,0), 2)
            out_vid.write(frame_full)

            n += 1
            if time.time() - fps_t0 > 2.0:
                fps = n / (time.time()-fps_t0)
                print(f"[FPS] {fps:.1f} | {text}")
                fps_t0, n = time.time(), 0

    except KeyboardInterrupt:
        pass
    finally:
        out_vid.release()
        cap.release()
        print("Saved: debug_out.mp4")

if __name__ == "__main__":
    main()
