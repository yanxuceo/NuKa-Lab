import { useMemo } from "react";
import { post } from "../api/nuka";
import type { NukaState } from "../state/nuka";

export default function Nuka({
  state,
  fullscreen = false,
}: {
  state: NukaState;
  fullscreen?: boolean;
}) {
  const done = state.current_reps >= state.target_reps;
  const pct = useMemo(() => {
    const t = Math.max(1, state.target_reps || 10);
    return Math.max(0, Math.min(100, (state.current_reps / t) * 100));
  }, [state.current_reps, state.target_reps]);

  return (
    <div style={fullscreen ? fsWrap() : wrap()}>
      <div style={header()}>
        <div style={{ fontSize: fullscreen ? 34 : 24, fontWeight: 900 }}>
          {state.active ? "⏰ Alarm ringing" : "Nuka Motion"}
        </div>
        <div style={{ opacity: 0.7, fontWeight: 700 }}>
          {Math.round(pct)}%
        </div>
      </div>

      <div style={fullscreen ? fsGrid() : grid()}>
        <div style={panel(fullscreen)}>
          <div style={{ opacity: 0.8, fontWeight: 700 }}>Squats</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 10 }}>
            <div style={{ fontSize: fullscreen ? 74 : 52, fontWeight: 950, lineHeight: 1 }}>
              {state.current_reps}
            </div>
            <div style={{ opacity: 0.55, fontSize: fullscreen ? 28 : 22, fontWeight: 800 }}>
              / {state.target_reps}
            </div>
          </div>

          <div style={{ marginTop: fullscreen ? 18 : 14, display: "flex", justifyContent: "center" }}>
            <Ring value={pct} size={fullscreen ? 220 : 160} />
          </div>

          <div style={{ display: "flex", gap: 12, marginTop: fullscreen ? 20 : 14 }}>
            <button style={btn()} onClick={() => post("/start")}>
              Start
            </button>

            <button
              style={btn(done ? "good" : "disabled")}
              disabled={!done}
              onClick={() => post("/stop")}
              title={done ? "Stop alarm" : "Finish reps to unlock"}
            >
              Stop
            </button>
          </div>

          {state.active && !done && (
            <div style={{ marginTop: 12, opacity: 0.75, fontSize: fullscreen ? 16 : 14 }}>
              Do squats to unlock Stop.
            </div>
          )}
        </div>

        {!fullscreen && (
          <div style={panel(false)}>
            <div style={{ fontWeight: 800, marginBottom: 10 }}>Quick test</div>
            <div style={{ display: "flex", gap: 10 }}>
              <button style={btn()} onClick={() => post("/rep")}>
                +1 Rep
              </button>
              <button style={btn("warn")} onClick={() => post("/start")}>
                Reset
              </button>
            </div>
            <div style={{ opacity: 0.65, marginTop: 10 }}>
              (Later driven by MoveNet /rep automatically.)
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function wrap(): React.CSSProperties {
  return { padding: 18, display: "grid", gap: 14 };
}
function fsWrap(): React.CSSProperties {
  return { padding: 22, height: "100%", display: "grid", gap: 18 };
}
function header(): React.CSSProperties {
  return { display: "flex", justifyContent: "space-between", alignItems: "baseline" };
}
function grid(): React.CSSProperties {
  return { display: "grid", gap: 14 };
}
function fsGrid(): React.CSSProperties {
  return { display: "grid", gap: 16, gridTemplateColumns: "1fr" };
}
function panel(fullscreen: boolean): React.CSSProperties {
  return {
    border: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(255,255,255,0.04)",
    borderRadius: 22,
    padding: fullscreen ? 18 : 14,
    boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
  };
}
function btn(kind?: "good" | "warn" | "disabled"): React.CSSProperties {
  const base: React.CSSProperties = {
    borderRadius: 16,
    padding: "12px 14px",
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(255,255,255,0.06)",
    color: "inherit",
    fontWeight: 800,
    cursor: "pointer",
    flex: 1,
    fontSize: 16,
  };
  if (kind === "good")
    return {
      ...base,
      background: "rgba(34,197,94,0.18)",
      border: "1px solid rgba(34,197,94,0.32)",
    };
  if (kind === "warn")
    return {
      ...base,
      background: "rgba(245,158,11,0.18)",
      border: "1px solid rgba(245,158,11,0.32)",
    };
  if (kind === "disabled") return { ...base, opacity: 0.35, cursor: "not-allowed" };
  return base;
}

function Ring({ value, size }: { value: number; size: number }) {
  const stroke = Math.max(10, Math.round(size * 0.08));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  const dash = (pct / 100) * c;

  return (
    <svg width={size} height={size} style={{ display: "block" }}>
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

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
        stroke="rgba(56,189,248,0.75)"
        strokeWidth={stroke}
        fill="transparent"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${c - dash}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        filter="url(#glow)"
      />
    </svg>
  );
}
