// src/App.tsx

import { BrowserRouter, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef} from "react";



import Home from "./pages/Home";
import Devices from "./pages/Devices";
import Nuka from "./pages/Nuka";
import { useNukaWS } from "./state/nuka";

function Shell() {
  const { data, connected } = useNukaWS("ws://localhost:8000/ws");
  const nav = useNavigate();
  const loc = useLocation();

  const prevActiveRef = useRef<boolean>(false);

  useEffect(() => {
    if (!data) return;

    const prev = prevActiveRef.current;
    const curr = !!data.active;

    // 只在 false -> true 的瞬间自动跳到 /nuka
    if (!prev && curr) {
      nav("/nuka");
    }

    prevActiveRef.current = curr;
  }, [data?.active, nav]);


    // Fullscreen alarm mode: hide dock
  if (data?.active) {
    return (
      <div style={fsRoot()}>
        <Nuka state={data} fullscreen />
        <div style={fsBadge()}>
          WS: {connected ? "online" : "reconnecting…"}
        </div>
      </div>
    );
  }

  return (
    <div style={root()}>
      <aside style={dock()}>
        <div style={{ fontWeight: 900, letterSpacing: 0.6, marginBottom: 14 }}>NuKa</div>

        <DockItem to="/" label="Home" />
        <DockItem to="/nuka" label="Nuka" />
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
