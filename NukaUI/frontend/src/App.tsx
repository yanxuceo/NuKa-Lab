// src/App.tsx

import { BrowserRouter, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { useCallback, useEffect, useRef, useState } from "react";



import Home from "./pages/Home";
import Devices from "./pages/Devices";
import Nuka from "./pages/Nuka";
import { useNukaWS } from "./state/nuka";
import type { NukaState } from "./state/nuka";

function Shell() {
  const { data, connected } = useNukaWS("ws://localhost:8000/ws");
  const nav = useNavigate();

  const prevActiveRef = useRef<boolean>(false);
  const lastActiveSnapshot = useRef<NukaState | null>(null);
  const [celebration, setCelebration] = useState<NukaState | null>(null);
  const celebrateTimerRef = useRef<number | null>(null);

  const stopCelebration = useCallback(
    (redirect: boolean) => {
      if (celebrateTimerRef.current) {
        window.clearTimeout(celebrateTimerRef.current);
        celebrateTimerRef.current = null;
      }
      setCelebration(null);
      if (redirect) {
        nav("/");
      }
    },
    [nav]
  );

  const startCelebration = useCallback(
    (snapshot: NukaState | null) => {
      if (!snapshot) return;
      if (celebrateTimerRef.current) {
        window.clearTimeout(celebrateTimerRef.current);
      }
      setCelebration(snapshot);
      celebrateTimerRef.current = window.setTimeout(() => {
        nav("/");
        // small delay keeps celebration screen during route change
        window.setTimeout(() => {
          setCelebration(null);
          celebrateTimerRef.current = null;
        }, 40);
      }, 4500);
    },
    [nav]
  );

  useEffect(() => {
    if (!data) {
      return;
    }
    if (data.active) {
      lastActiveSnapshot.current = data;
      const doneNow =
        data.state === "DONE" ||
        data.current_reps >= (data.target_reps || 10);
      if (doneNow && !celebration) {
        startCelebration(data);
      }
    }
  }, [data, celebration, startCelebration]);

  useEffect(() => {
    if (!data) return;

    const prev = prevActiveRef.current;
    const curr = !!data.active;

    // false -> true：进入 Nuka
    if (!prev && curr) {
      stopCelebration(false);
      nav("/nuka");
    }

    // true -> false：完成或回 Home
    if (prev && !curr) {
      const snap = lastActiveSnapshot.current;
      const cleared = snap && snap.current_reps >= (snap.target_reps || 10);
      if (cleared && snap) {
        if (!celebration) {
          startCelebration(snap);
        }
      } else {
        stopCelebration(true);
      }
    }

    prevActiveRef.current = curr;
  }, [data, nav, celebration, startCelebration, stopCelebration]);

  useEffect(() => {
    return () => {
      if (celebrateTimerRef.current) {
        window.clearTimeout(celebrateTimerRef.current);
      }
    };
  }, []);

  const displayState = celebration ?? (data?.active ? data : null);
  const fullscreenMode = !!displayState;

  // Fullscreen alarm mode: hide dock
  if (fullscreenMode && displayState) {
    return (
      <div style={fsRoot()}>
        <Nuka state={displayState} fullscreen celebrationMode={!!celebration} />
        <div style={fsBadge()}>
          WS: {connected ? "online" : "reconnecting…"}
        </div>
      </div>
    );
  }

  return (
    <div style={root()}>
      <aside style={dock()}>
        <div style={{ fontWeight: 900, letterSpacing: 0.6, marginBottom: 14 }}>NuKa Lab</div>

        <DockItem to="/" label="Home" />
        <DockItem to="/nuka" label="Motion" />
        <DockItem to="/devices" label="Devices" />

        <div style={{ marginTop: "auto", opacity: 0.7, fontSize: 12 }}>
          WS: {connected ? "online" : "reconnecting…"}
        </div>
      </aside>

      <main style={main()}>
        {!data ? (
          <div style={{ padding: 18 }}>Connecting to Nuka…</div>
        ) : (
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/devices" element={<Devices />} />
            <Route path="/nuka" element={<Nuka state={data} />} />
            <Route path="*" element={<Home />} />
          </Routes>
        )}
      </main>
    </div>
  );

}

export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}

function DockItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      style={({ isActive }) => ({
        textDecoration: "none",
        color: "inherit",
        display: "block",
        padding: "10px 12px",
        borderRadius: 14,
        marginBottom: 10,
        background: isActive ? "rgba(56,189,248,0.18)" : "transparent",
        border: isActive
          ? "1px solid rgba(56,189,248,0.30)"
          : "1px solid rgba(255,255,255,0.10)",
      })}
    >
      {label}
    </NavLink>
  );
}

function root(): React.CSSProperties {
  return {
    height: "100vh",
    background: "#0b0f14",
    color: "#e5e7eb",
    display: "grid",
    gridTemplateColumns: "140px 1fr",
    fontFamily: "system-ui",
  };
}

function dock(): React.CSSProperties {
  return {
    padding: 14,
    borderRight: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(255,255,255,0.02)",
    display: "flex",
    flexDirection: "column",
  };
}

function main(): React.CSSProperties {
  return { overflow: "auto" };
}


function fsRoot(): React.CSSProperties {
  return {
    height: "100vh",
    background: "#0b0f14",
    color: "#e5e7eb",
    fontFamily: "system-ui",
    position: "relative",
  };
}

function fsBadge(): React.CSSProperties {
  return {
    position: "absolute",
    right: 12,
    bottom: 10,
    fontSize: 12,
    opacity: 0.65,
  };
}
