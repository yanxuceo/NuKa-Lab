import { useEffect, useState } from "react";

export type NukaState = {
  active: boolean;
  state?: string; // future: RINGING/SQUAT_ACTIVE/UNLOCKED
  current_reps: number;
  target_reps: number;
};

export function useNukaWS(url = "ws://localhost:8000/ws") {
  const [data, setData] = useState<NukaState | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let alive = true;

    const connect = () => {
      ws = new WebSocket(url);

      ws.onopen = () => {
        if (!alive) return;
        setConnected(true);
      };

      ws.onclose = () => {
        if (!alive) return;
        setConnected(false);
        // simple reconnect
        setTimeout(connect, 800);
      };

      ws.onerror = () => {
        // onclose will handle reconnect
      };

      ws.onmessage = (evt) => {
        try {
          const obj = JSON.parse(evt.data);
          setData(obj);
        } catch {
          // ignore
        }
      };
    };

    connect();
    return () => {
      alive = false;
      try {
        ws?.close();
      } catch {}
    };
  }, [url]);

  return { data, connected };
}
