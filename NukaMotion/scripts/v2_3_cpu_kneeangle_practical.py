import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "movenet_lightning_int8.tflite"
CAM_INDEX = 0

L_SHOULDER=5; R_SHOULDER=6
L_HIP=11; R_HIP=12
L_KNEE=13; R_KNEE=14
L_ANKLE=15; R_ANKLE=16

SKELETON = [
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, L_KNEE), (R_HIP, R_KNEE),
    (L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE),
]

def letterbox_rgb_u8(frame_bgr, out_size=192):
    h, w = frame_bgr.shape[:2]
    scale = out_size / max(h, w)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_left = (out_size - nw)//2
    pad_top  = (out_size - nh)//2
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, (scale, pad_left, pad_top)

def kp_to_orig_xy(kp_y, kp_x, meta, orig_w, orig_h, in_size=192):
    scale, pad_left, pad_top = meta
    x = (kp_x * in_size - pad_left) / scale
    y = (kp_y * in_size - pad_top) / scale
    x = float(np.clip(x, 0, orig_w-1))
    y = float(np.clip(y, 0, orig_h-1))
    return (x, y)

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
    """
    目标：少漏检（demo优先），在丢帧/低置信度时也能“续命”
    """
    def __init__(self,
                 down_th=120.0,
                 up_th=155.0,
                 min_score=0.15,
                 cooldown=0.15,
                 confirm_frames=1):
        self.state = "STAND"
        self.count = 0
        self.down_th = down_th
        self.up_th = up_th
        self.min_score = min_score
        self.cooldown = cooldown
        self.confirm_frames = confirm_frames

        self.last_t = 0.0
        self.ema = EMA(alpha=0.25)
        self.down_hits = 0
        self.up_hits = 0
        self.last_valid_ang = None
        self.last_valid_t = 0.0

    def _leg_angle(self, kps, meta, w, h, hip_i, knee_i, ankle_i):
        hy,hx,hs = kps[hip_i]
        ky,kx,ks = kps[knee_i]
        ay,ax,as_ = kps[ankle_i]
        if min(hs, ks, as_) < self.min_score:
            return None, None
        hip = kp_to_orig_xy(hy,hx,meta,w,h)
        knee = kp_to_orig_xy(ky,kx,meta,w,h)
        ankle = kp_to_orig_xy(ay,ax,meta,w,h)
        return angle_3pts(hip, knee, ankle), (hip,knee,ankle)

    def compute_angle(self, kps, meta, w, h):
        la, lpts = self._leg_angle(kps, meta, w, h, L_HIP, L_KNEE, L_ANKLE)
        ra, rpts = self._leg_angle(kps, meta, w, h, R_HIP, R_KNEE, R_ANKLE)

        if la is None and ra is None:
            return None, None, "no_leg"
        if la is None:
            ang, pts, src = ra, {"R": rpts}, "R"
        elif ra is None:
            ang, pts, src = la, {"L": lpts}, "L"
        else:
            ang, pts, src = (la+ra)/2.0, {"L": lpts, "R": rpts}, "LR"

        # 只过滤特别离谱的值，避免漏检
        if not (30.0 <= ang <= 180.0):
            ang = None

        ang = self.ema.update(ang)

        # 续命：若本帧算不出角度，就用最近 0.6s 内的角度顶一下
        now = time.time()
        if ang is None and self.last_valid_ang is not None and (now - self.last_valid_t) < 0.6:
            ang = self.last_valid_ang
            src = src + "+hold"
        elif ang is not None:
            self.last_valid_ang = ang
            self.last_valid_t = now

        return ang, pts, src

    def update(self, ang):
        if ang is None:
            return None

        now = time.time()
        if now - self.last_t < self.cooldown:
            return ang, None, self.count

        event = None
        if self.state == "STAND":
            self.up_hits = 0
            if ang < self.down_th:
                self.down_hits += 1
                if self.down_hits >= self.confirm_frames:
                    self.state = "DOWN"
                    self.last_t = now
                    self.down_hits = 0
                    event = "DOWN"
            else:
                self.down_hits = 0
        else:  # DOWN
            self.down_hits = 0
            if ang > self.up_th:
                self.up_hits += 1
                if self.up_hits >= self.confirm_frames:
                    self.state = "STAND"
                    self.last_t = now
                    self.up_hits = 0
                    self.count += 1
                    event = "SQUAT_DONE"
            else:
                self.up_hits = 0

        return ang, event, self.count

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

    fps_t0, n = time.time(), 0
    print("Running V2.3 (practical)... Ctrl+C to stop -> debug_out.mp4")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            rgb, meta = letterbox_rgb_u8(frame, 192)
            x = rgb[np.newaxis, ...].astype(np.uint8)

            t0 = time.time()
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()
            kps = interpreter.get_tensor(out["index"])[0, 0, :, :]
            infer_ms = (time.time()-t0)*1000.0

            scores = kps[:,2]
            min_s = float(scores.min()); mean_s = float(scores.mean())

            ang, pts, src = det.compute_angle(kps, meta, w, h)
            res = det.update(ang)

            if res is None:
                text = f"KP low | minS={min_s:.2f} meanS={mean_s:.2f} | {infer_ms:.1f}ms"
            else:
                ang, event, cnt = res
                text = f"knee={ang:.1f} src={src} state={det.state} cnt={cnt} | minS={min_s:.2f} meanS={mean_s:.2f} | {infer_ms:.1f}ms"
                if event == "DOWN":
                    print("⬇️ DOWN")
                elif event == "SQUAT_DONE":
                    print("✅ SQUAT_DONE | cnt=", cnt)

            # 绘制（尽量画，便于调试）
            min_draw = 0.15
            draw_pts = {}
            for idx in [L_SHOULDER,R_SHOULDER,L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE]:
                ky,kx,ks = kps[idx]
                if ks >= min_draw:
                    ox, oy = kp_to_orig_xy(ky, kx, meta, w, h, 192)
                    draw_pts[idx] = (int(ox), int(oy))
                    cv2.circle(frame, draw_pts[idx], 4, (0,255,0), -1)
            for a,b in SKELETON:
                if a in draw_pts and b in draw_pts:
                    cv2.line(frame, draw_pts[a], draw_pts[b], (0,255,0), 2)

            cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            out_vid.write(frame)

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
