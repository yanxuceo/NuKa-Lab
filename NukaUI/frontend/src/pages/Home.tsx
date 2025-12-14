import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

function pad2(n: number) {
  return n < 10 ? `0${n}` : `${n}`;
}

function formatTime(d: Date) {
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

function formatDate(d: Date) {
  const w = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getDay()];
  return `${w}  ${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

export default function Home() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 500);
    return () => clearInterval(t);
  }, []);

  const timeText = useMemo(() => formatTime(now), [now]);
  const dateText = useMemo(() => formatDate(now), [now]);

  return (
    <div style={wrap()}>
      {/* Top hero */}
      <div style={hero()}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 68, fontWeight: 950, letterSpacing: -1.2, lineHeight: 1 }}>
            {timeText}
          </div>
          <div style={{ fontSize: 16, opacity: 0.75, fontWeight: 700 }}>{dateText}</div>
        </div>

        <div style={pillRow()}>
          <Pill label="System" value="Online" />
          <Pill label="Mode" value="Home" />
          <Pill label="Network" value="LAN" />
        </div>
      </div>

      {/* Cards */}
      <div style={grid()}>
        <Link to="/nuka" style={{ textDecoration: "none", color: "inherit" }}>
          <Card
            title="Nuka Motion"
            desc="Squat alarm & rep counter"
            accent="cyan"
            right={
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 14, opacity: 0.75, fontWeight: 700 }}>Quick</div>
                <div style={{ fontSize: 22, fontWeight: 900 }}>Open</div>
              </div>
            }
          />
        </Link>

        <Link to="/devices" style={{ textDecoration: "none", color: "inherit" }}>
          <Card
            title="Devices"
            desc="ESP32 nodes & sensors"
            accent="violet"
            right={
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 14, opacity: 0.75, fontWeight: 700 }}>Soon</div>
                <div style={{ fontSize: 22, fontWeight: 900 }}>MQTT</div>
              </div>
            }
          />
        </Link>

        <Card
          title="Today"
          desc="Light overview (placeholder)"
          accent="amber"
          right={
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 14, opacity: 0.75, fontWeight: 700 }}>Focus</div>
              <div style={{ fontSize: 22, fontWeight: 900 }}>Move</div>
            </div>
          }
        />

        <Card
          title="Status"
          desc="WS / services (placeholder)"
          accent="green"
          right={
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 14, opacity: 0.75, fontWeight: 700 }}>Backend</div>
              <div style={{ fontSize: 22, fontWeight: 900 }}>OK</div>
            </div>
          }
        />
      </div>

      {/* Footer hint */}
      <div style={{ opacity: 0.65, fontSize: 12, paddingLeft: 2 }}>
        Tip: When alarm rings, it auto-switches to full-screen Nuka.
      </div>
    </div>
  );
}

function wrap(): React.CSSProperties {
  return {
    padding: 18,
    height: "100%",
    display: "grid",
    gap: 14,
    // brighter, airy background with subtle tech gradient
    background:
      "radial-gradient(1200px 700px at 20% -10%, rgba(56,189,248,0.35), transparent 60%)," +
      "radial-gradient(900px 600px at 95% 10%, rgba(168,85,247,0.28), transparent 55%)," +
      "radial-gradient(900px 700px at 30% 115%, rgba(34,197,94,0.18), transparent 55%)," +
      "linear-gradient(180deg, rgba(248,250,252,0.10), rgba(255,255,255,0.02))",
    borderRadius: 18,
  };
}

function hero(): React.CSSProperties {
  return {
    borderRadius: 24,
    padding: 18,
    border: "1px solid rgba(255,255,255,0.14)",
    background:
      "linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06))",
    boxShadow: "0 18px 45px rgba(0,0,0,0.18)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 14,
  };
}

function pillRow(): React.CSSProperties {
  return {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    alignItems: "flex-end",
    minWidth: 140,
  };
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        borderRadius: 999,
        padding: "8px 12px",
        border: "1px solid rgba(255,255,255,0.14)",
        background: "rgba(255,255,255,0.10)",
        display: "flex",
        gap: 10,
        alignItems: "baseline",
        justifyContent: "space-between",
        width: 150,
      }}
    >
      <span style={{ fontSize: 12, opacity: 0.7, fontWeight: 800 }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 900 }}>{value}</span>
    </div>
  );
}

function grid(): React.CSSProperties {
  return {
    display: "grid",
    gap: 12,
    gridTemplateColumns: "1fr 1fr",
  };
}

function Card({
  title,
  desc,
  accent,
  right,
}: {
  title: string;
  desc: string;
  accent: "cyan" | "violet" | "amber" | "green";
  right?: React.ReactNode;
}) {
  const accentMap: Record<string, string> = {
    cyan: "rgba(56,189,248,0.55)",
    violet: "rgba(168,85,247,0.45)",
    amber: "rgba(245,158,11,0.35)",
    green: "rgba(34,197,94,0.35)",
  };

  return (
    <div
      style={{
        borderRadius: 22,
        padding: 14,
        border: "1px solid rgba(255,255,255,0.14)",
        background:
          "linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.05))",
        boxShadow: "0 16px 40px rgba(0,0,0,0.14)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* accent glow */}
      <div
        style={{
          position: "absolute",
          inset: -60,
          background: `radial-gradient(circle at 30% 30%, ${accentMap[accent]}, transparent 45%)`,
          filter: "blur(10px)",
          opacity: 0.9,
          pointerEvents: "none",
        }}
      />

      <div style={{ position: "relative" }}>
        <div style={{ fontSize: 18, fontWeight: 950 }}>{title}</div>
        <div style={{ opacity: 0.72, marginTop: 6, fontSize: 13, fontWeight: 650 }}>
          {desc}
        </div>
      </div>

      <div style={{ position: "relative" }}>{right}</div>
    </div>
  );
}
