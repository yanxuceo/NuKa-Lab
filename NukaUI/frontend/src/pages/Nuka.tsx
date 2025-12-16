import { useMemo } from "react";
import type { NukaState } from "../state/nuka";

export default function Nuka({
  state,
  fullscreen = false,
}: {
  state: NukaState;
  fullscreen?: boolean;
}) {
  const done =
    state.state === "DONE" || state.current_reps >= (state.target_reps || 10);

  const pct = useMemo(() => {
    const t = Math.max(1, state.target_reps || 10);
    return Math.max(0, Math.min(100, (state.current_reps / t) * 100));
  }, [state.current_reps, state.target_reps]);

  const title = done
    ? "✅ Completed"
    : state.active
      ? "⏰ Alarm ringing"
      : "Nuka Motion";

  return (
    <div style={fullscreen ? fsWrap() : wrap()}>
      <div style={header()}>
        <div style={{ fontSize: fullscreen ? 34 : 24, fontWeight: 900 }}>
          {title}
        </div>
        <div style={{ opacity: 0.7, fontWeight: 700 }}>{Math.round(pct)}%</div>
      </div>

      <div style={fullscreen ? fsGrid() : grid()}>
        <div style={panel(fullscreen)}>
          <div style={{ opacity: 0.8, fontWeight: 700 }}>
            {done ? "Pump up" : "Squats"}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: 10,
              marginTop: 10,
            }}
          >
            <div
              style={{
                fontSize: fullscreen ? 74 : 52,
                fontWeight: 950,
                lineHeight: 1,
              }}
            >
              {state.current_reps}
            </div>
            <div
              style={{
                opacity: 0.55,
                fontSize: fullscreen ? 28 : 22,
                fontWeight: 800,
              }}
            >
              / {state.target_reps}
            </div>
          </div>

          <div
            style={{
              marginTop: fullscreen ? 18 : 14,
              display: "flex",
              justifyContent: "center",
            }}
          >
            {done ? (
              <CelebrateRing
                value={pct}
                size={fullscreen ? 240 : 180}
                label="PUMPED"
              />
            ) : (
              <Ring value={pct} size={fullscreen ? 220 : 160} />
            )}
          </div>

          {state.active && !done && (
            <div
              style={{
                marginTop: 14,
                opacity: 0.75,
                fontSize: fullscreen ? 16 : 14,
              }}
            >
              Do squats to unlock completion.
            </div>
          )}

          {done && (
            <div
              style={{
                marginTop: 14,
                display: "grid",
                gap: 8,
              }}
            >
              <div
                style={{
                  fontWeight: 900,
                  fontSize: fullscreen ? 20 : 18,
                }}
              >
                Nice work!
              </div>
              <div style={{ opacity: 0.75, fontSize: fullscreen ? 16 : 14 }}>
                Breathing… heart rate up… returning to Home.
              </div>
            </div>
          )}
        </div>
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
  return {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "baseline",
  };
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

  // CSS-only animations (cheap on Jetson)
  const pulse = {
    animation: "nukaPulse 1.1s ease-in-out infinite",
  } as const;

  const breathe = {
    animation: "nukaBreathe 2.2s ease-in-out infinite",
  } as const;

  const shimmer = {
    animation: "nukaShimmer 1.6s linear infinite",
  } as const;

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <style>{cssKeyframes()}</style>

      <svg width={size} height={size} style={{ display: "block" }}>
        <defs>
          <filter id="glow2">
            <feGaussianBlur stdDeviation="3.2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="rgba(255,255,255,0.10)"
          strokeWidth={stroke}
          fill="transparent"
        />

        {/* breathing halo */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="rgba(56,189,248,0.18)"
          strokeWidth={stroke * 1.35}
          fill="transparent"
          style={breathe}
        />

        {/* progress ring (kept) */}
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
          filter="url(#glow2)"
          style={shimmer}
        />

        {/* heartbeat dot */}
        <circle
          cx={size / 2}
          cy={size / 2 - r}
          r={Math.max(4, Math.round(stroke * 0.35))}
          fill="rgba(56,189,248,0.95)"
          style={pulse}
        />
      </svg>

      <div style={centerLabel()}>
        <div style={{ fontWeight: 950, letterSpacing: 1.6 }}>{label}</div>
        <div style={{ opacity: 0.7, fontSize: 12 }}>Great job</div>
      </div>
    </div>
  );
}

function centerLabel(): React.CSSProperties {
  return {
    position: "absolute",
    inset: 0,
    display: "grid",
    placeItems: "center",
    textAlign: "center",
    pointerEvents: "none",
  };
}

function cssKeyframes() {
  return `
@keyframes nukaPulse {
  0%   { transform: scale(1); opacity: 0.85; }
  45%  { transform: scale(1.55); opacity: 1; }
  100% { transform: scale(1); opacity: 0.85; }
}

@keyframes nukaBreathe {
  0%   { opacity: 0.20; transform: scale(1); }
  50%  { opacity: 0.55; transform: scale(1.06); }
  100% { opacity: 0.20; transform: scale(1); }
}

@keyframes nukaShimmer {
  0%   { opacity: 0.70; }
  50%  { opacity: 1.00; }
  100% { opacity: 0.70; }
}
`;
}
