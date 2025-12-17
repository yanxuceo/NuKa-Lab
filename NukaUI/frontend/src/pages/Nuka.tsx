import { useMemo } from "react";
import type { NukaState } from "../state/nuka";

const MJPEG_URL = "http://localhost:8000/mjpeg";

export default function Nuka({
  state,
  fullscreen = false,
  celebrationMode = false,
}: {
  state: NukaState;
  fullscreen?: boolean;
  celebrationMode?: boolean;
}) {
  const done =
    state.state === "DONE" || state.current_reps >= (state.target_reps || 10);

  const pct = useMemo(() => {
    const t = Math.max(1, state.target_reps || 10);
    return Math.max(0, Math.min(100, (state.current_reps / t) * 100));
  }, [state.current_reps, state.target_reps]);

  const confettiPieces = useMemo(() => {
    if (!celebrationMode) {
      return [];
    }
    const colors = ["#38bdf8", "#a855f7", "#facc15", "#f97316"];
    return Array.from({ length: 16 }).map(() => ({
      left: Math.random() * 100,
      top: Math.random() * 100,
      size: 6 + Math.random() * 18,
      opacity: 0.35 + Math.random() * 0.45,
      color: colors[Math.floor(Math.random() * colors.length)],
      rotate: Math.random() * 360,
    }));
  }, [celebrationMode]);

  return (
    <div style={wrap(fullscreen)}>
      {/* ===== ACTIVE / IN-PROGRESS ===== */}
      {!done && (
        <div style={activeGrid(fullscreen)}>
          {/* LEFT: Ring / Counter */}
          <div style={leftPanel()}>
            <div style={title()}>Squats</div>

            <div style={counter()}>
              <span style={countNum(fullscreen)}>{state.current_reps}</span>
              <span style={countSep()}>/</span>
              <span style={countTotal()}>{state.target_reps}</span>
            </div>

            <Ring value={pct} size={fullscreen ? 220 : 180} />

            <div style={hint()}>Do squats in front of the camera</div>
          </div>

          {/* RIGHT: Video as background */}
          <div style={videoPanel(fullscreen)}>
            <div style={videoWrap()}>
              <div style={videoFrame(fullscreen)}>
                <img src={`${MJPEG_URL}?cache=1`} style={video()} />
                {/* subtle HUD overlay */}
                <div style={videoMask()} />
                <div style={videoBadge()}>Live camera</div>
              </div>
              <div style={videoMetaRow()}>
                <VideoMeta label="Perspective" value="Side view" />
                <VideoMeta label="Tracking" value="ROI auto" />
                <VideoMeta label="FPS" value="~20" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== DONE / CELEBRATION ===== */}
      {done && (
        <div style={doneWrap(celebrationMode)}>
          {celebrationMode && <Confetti pieces={confettiPieces} />}
          <div style={badgeRow()}>
            <span style={badgePill("rgba(56,189,248,0.8)")}>ALARM CLEARED</span>
            <span style={badgePill("rgba(168,85,247,0.8)")}>SQUAT POWER</span>
          </div>

          <CelebrateRing
            value={pct}
            size={fullscreen ? 260 : 220}
            label="PUMPED"
          />

          <div style={doneText(fullscreen)}>Mission accomplished</div>
          <div style={doneSub()}>
            Breathing steady. Nervous system awake.
          </div>

          <div style={metricsRow()}>
            <Metric label="Reps" value={`${state.current_reps}`} />
            <Metric label="Target" value={`${state.target_reps}`} />
            <Metric label="Mood" value="AWAKE" />
          </div>
        </div>
      )}
    </div>
  );
}

/* =========================
   Layout styles
========================= */

function wrap(fullscreen: boolean): React.CSSProperties {
  return {
    height: "100%",
    width: "100%",
    padding: fullscreen ? 18 : 14,
    boxSizing: "border-box",
  };
}

function activeGrid(fullscreen: boolean): React.CSSProperties {
  return {
    height: "100%",
    display: "grid",
    gridTemplateColumns: fullscreen ? "420px 1fr" : "360px 1fr",
    gap: 18,
  };
}

function leftPanel(): React.CSSProperties {
  return {
    borderRadius: 22,
    padding: 18,
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.10)",
    display: "flex",
    flexDirection: "column",
    gap: 18,
    alignItems: "center",
    textAlign: "center",
    justifyContent: "center",
  };
}

function videoPanel(fullscreen: boolean): React.CSSProperties {
  return {
    position: "relative",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: fullscreen ? 18 : 14,
    borderRadius: 22,
    background:
      "radial-gradient(circle at 20% 20%, rgba(255,255,255,0.08), transparent 55%), rgba(3,5,8,0.92)",
    border: "1px solid rgba(255,255,255,0.08)",
    minHeight: "100%",
  };
}

function videoWrap(): React.CSSProperties {
  return {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    width: "100%",
    alignItems: "center",
  };
}

function videoFrame(fullscreen: boolean): React.CSSProperties {
  return {
    position: "relative",
    width: "100%",
    maxWidth: fullscreen ? 1280 : 920,
    aspectRatio: fullscreen ? "16 / 10" : "16 / 10",
    borderRadius: 24,
    overflow: "hidden",
    boxShadow: "0 25px 45px rgba(0,0,0,0.45)",
    border: "1px solid rgba(255,255,255,0.10)",
    background: "#000",
  };
}

function video(): React.CSSProperties {
  return {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
  };
}

function videoMask(): React.CSSProperties {
  return {
    position: "absolute",
    inset: 0,
    background:
      "linear-gradient(120deg, rgba(11,15,20,0.65) 0%, rgba(11,15,20,0.10) 40%, rgba(11,15,20,0.00) 100%)",
    pointerEvents: "none",
  };
}

function videoBadge(): React.CSSProperties {
  return {
    position: "absolute",
    left: 14,
    top: 12,
    padding: "4px 12px",
    borderRadius: 999,
    fontSize: 12,
    letterSpacing: 0.6,
    fontWeight: 700,
    background: "rgba(56,189,248,0.85)",
    color: "#02131f",
  };
}

function videoMetaRow(): React.CSSProperties {
  return {
    display: "flex",
    gap: 14,
    flexWrap: "wrap",
    justifyContent: "center",
  };
}

function videoMetaCard(): React.CSSProperties {
  return {
    minWidth: 130,
    padding: "10px 12px",
    borderRadius: 16,
    background: "rgba(0,0,0,0.35)",
    border: "1px solid rgba(255,255,255,0.1)",
    textAlign: "center",
  };
}

function VideoMeta({ label, value }: { label: string; value: string }) {
  return (
    <div style={videoMetaCard()}>
      <div style={{ fontSize: 11, opacity: 0.65, letterSpacing: 0.8, fontWeight: 700 }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 800 }}>{value}</div>
    </div>
  );
}

/* =========================
   Text & counters
========================= */

function title(): React.CSSProperties {
  return {
    fontSize: 20,
    fontWeight: 900,
    opacity: 0.85,
  };
}

function counter(): React.CSSProperties {
  return {
    display: "flex",
    alignItems: "baseline",
    gap: 6,
    justifyContent: "center",
  };
}

function countNum(full: boolean): React.CSSProperties {
  return {
    fontSize: full ? 64 : 56,
    fontWeight: 950,
    lineHeight: 1,
  };
}

function countSep(): React.CSSProperties {
  return {
    fontSize: 28,
    opacity: 0.5,
  };
}

function countTotal(): React.CSSProperties {
  return {
    fontSize: 26,
    opacity: 0.65,
    fontWeight: 800,
  };
}

function hint(): React.CSSProperties {
  return {
    marginTop: 8,
    fontSize: 14,
    opacity: 0.65,
    textAlign: "center",
  };
}

/* =========================
   DONE state
========================= */

function doneWrap(celebrating: boolean): React.CSSProperties {
  return {
    height: "100%",
    display: "grid",
    placeItems: "center",
    alignContent: "center",
    gap: 18,
    position: "relative",
    overflow: "hidden",
    padding: celebrating ? 28 : 18,
    borderRadius: celebrating ? 28 : 18,
    background: celebrating
      ? "radial-gradient(circle at 20% 20%, rgba(56,189,248,0.45), transparent 60%), radial-gradient(circle at 80% 0%, rgba(168,85,247,0.35), transparent 55%), rgba(10,15,20,0.92)"
      : "rgba(10,15,20,0.85)",
    border: "1px solid rgba(255,255,255,0.10)",
    boxShadow: celebrating ? "0 30px 60px rgba(0,0,0,0.35)" : undefined,
  };
}

function doneText(full: boolean): React.CSSProperties {
  return {
    fontSize: full ? 28 : 22,
    fontWeight: 900,
  };
}

function doneSub(): React.CSSProperties {
  return {
    fontSize: 14,
    opacity: 0.65,
    textAlign: "center",
  };
}

function badgeRow(): React.CSSProperties {
  return {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
    justifyContent: "center",
  };
}

function badgePill(color: string): React.CSSProperties {
  return {
    borderRadius: 999,
    padding: "6px 14px",
    fontSize: 12,
    fontWeight: 800,
    letterSpacing: 0.8,
    background: color,
    color: "#0b0f14",
  };
}

function metricsRow(): React.CSSProperties {
  return {
    display: "flex",
    gap: 14,
    flexWrap: "wrap",
    justifyContent: "center",
  };
}

function metricCard(): React.CSSProperties {
  return {
    minWidth: 110,
    padding: "10px 14px",
    borderRadius: 18,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.10)",
    textAlign: "center",
  };
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={metricCard()}>
      <div style={{ fontSize: 11, opacity: 0.65, letterSpacing: 0.8, fontWeight: 700 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 900 }}>{value}</div>
    </div>
  );
}

type ConfettiPiece = {
  left: number;
  top: number;
  size: number;
  opacity: number;
  color: string;
  rotate: number;
};

function Confetti({ pieces }: { pieces: ConfettiPiece[] }) {
  return (
    <div style={confettiLayer()}>
      {pieces.map((p, idx) => (
        <span
          key={idx}
          style={{
            position: "absolute",
            left: `${p.left}%`,
            top: `${p.top}%`,
            width: p.size,
            height: p.size * 3,
            background: p.color,
            opacity: p.opacity,
            transform: `rotate(${p.rotate}deg)`,
            borderRadius: 6,
            filter: "blur(0.2px)",
          }}
        />
      ))}
    </div>
  );
}

function confettiLayer(): React.CSSProperties {
  return {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    overflow: "hidden",
  };
}

/* =========================
   Rings (unchanged logic)
========================= */

function Ring({ value, size }: { value: number; size: number }) {
  const stroke = Math.max(10, Math.round(size * 0.08));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  const dash = (pct / 100) * c;

  return (
    <svg width={size} height={size}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        stroke="rgba(255,255,255,0.10)"
        strokeWidth={stroke}
        fill="transparent"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        stroke="rgba(56,189,248,0.85)"
        strokeWidth={stroke}
        fill="transparent"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${c - dash}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
    </svg>
  );
}

/* =========================
   Celebrate ring (原样保留)
========================= */

function CelebrateRing({
  value,
  size,
  label,
}: {
  value: number;
  size: number;
  label: string;
}) {
  const stroke = Math.max(10, Math.round(size * 0.085));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  const dash = (pct / 100) * c;

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={stroke}
          fill="transparent"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="rgba(56,189,248,0.9)"
          strokeWidth={stroke}
          fill="transparent"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c - dash}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>

      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "grid",
          placeItems: "center",
          fontWeight: 950,
          letterSpacing: 1.6,
        }}
      >
        {label}
      </div>
    </div>
  );
}
