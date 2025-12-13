import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "movenet_lightning_int8.tflite"
CAM_INDEX = 0

# COCO17 index
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
    return rgb, (scale, pad_left, pad_top, nw, nh)

def kp_to_orig_xy(kp_y, kp_x, meta, orig_w, orig_h, in_size=192):
    scale, pad_left, pad_top, nw, nh = meta
    x = kp_x * in_size - pad_left
    y = kp_y * in_size - pad_top
    x /= scale; y /= scale
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
            return None
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha*x + (1-self.alpha)*self.v
        return float(self.v)

class SquatDetectorByKneeAngle:
    """
    - 只要左腿或右腿有一侧可用，就计算膝角（适合侧身/遮挡）
    """
    def __init__(self, down_th=120.0, up_th=155.0, min_score=0.15, cooldown=0.25):
        self.state = "STAND"
        self.count = 0
        self.down_th = float(down_th)
        self.up_th = float(up_th)
        self.min_score = float(min_score)
        self.cooldown = float(cooldown)
        self.last_t = 0.0
        self.ema = EMA(alpha=0.25)

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

    def compute_knee_angle(self, kps, meta, w, h):
        left_ang, left_pts = self._leg_angle(kps, meta, w, h, L_HIP, L_KNEE, L_ANKLE)
        right_ang, right_pts = self._leg_angle(kps, meta, w, h, R_HIP, R_KNEE, R_ANKLE)

        # 选择更可信的一侧：这里简单用“有值优先；两边都有取平均”
        if left_ang is None and right_ang is None:
            return None, None, "no_leg"
        if left_ang is None:
            ang = right_ang
            pts = {"R": right_pts}
            src = "R"
        elif right_ang is None:
            ang = left_ang
            pts = {"L": left_pts}
            src = "L"
        else:
            ang = (left_ang + right_ang) / 2.0
            pts = {"L": left_pts, "R": right_pts}
            src = "LR"

        ang = self.ema.update(ang)
        return ang, pts, src

    def update(self, knee_angle):
        if knee_angle is None:
            return None

        now = time.time()
        if now - self.last_t < self.cooldown:
            return knee_angle, None, self.count

        event = None
        if self.state == "STAND" and knee_angle < self.down_th:
            self.state = "DOWN"
            self.last_t = now
            event = "DOWN"
        elif self.state == "DOWN" and knee_angle > self.up_th:
            self.state = "STAND"
            self.last_t = now
            self.count += 1
            event = "SQUAT_DONE"

        return knee_angle, event, self.count

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

    detector = SquatDetectorByKneeAngle()

    fps_t0, n = time.time(), 0
    print("Running V2.1 (knee-angle debug)... Ctrl+C to stop -> debug_out.mp4")

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
            kps = interpreter.get_tensor(out["index"])[0, 0, :, :]  # [17,3]
            infer_ms = (time.time() - t0) * 1000.0

            # 统计一下关键点整体最低/平均分（诊断用）
            scores = kps[:,2]
            min_s = float(scores.min())
            mean_s = float(scores.mean())

            knee_angle, leg_pts, src = detector.compute_knee_angle(kps, meta, w, h)
            res = detector.update(knee_angle)

            if res is None:
                text = f"KP low conf | minS={min_s:.2f} meanS={mean_s:.2f} | infer={infer_ms:.1f}ms"
            else:
                ang, event, cnt = res
                text = f"knee={ang:.1f}deg src={src} state={detector.state} cnt={cnt} | minS={min_s:.2f} meanS={mean_s:.2f} | {infer_ms:.1f}ms"
                if event == "SQUAT_DONE":
                    print("✅ SQUAT_DONE -> stop alarm here | cnt=", cnt)
                elif event == "DOWN":
                    print("⬇️ DOWN")

            # 画骨架：即使分数低也尽量画（帮助你理解“为什么低”）
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

            # 额外把当前使用的腿（L/R）标一下
            if leg_pts is not None:
                if "L" in leg_pts:
                    hip,knee,ankle = leg_pts["L"]
                    cv2.putText(frame, "L", (int(knee[0])+8, int(knee[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                if "R" in leg_pts:
                    hip,knee,ankle = leg_pts["R"]
                    cv2.putText(frame, "R", (int(knee[0])+8, int(knee[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            out_vid.write(frame)

            n += 1
            if time.time() - fps_t0 > 2.0:
                fps = n / (time.time() - fps_t0)
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
