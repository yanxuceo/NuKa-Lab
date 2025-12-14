// src/pages/Devices.tsx

function Devices() {
  return (
    <div style={{ padding: 18 }}>
      <div style={{ fontSize: 24, fontWeight: 800, marginBottom: 10 }}>
        Devices
      </div>
      <div style={{ opacity: 0.75 }}>
        Next step: connect MQTT and show ESP32 nodes here.
      </div>

      <div
        style={{
          marginTop: 14,
          border: "1px solid rgba(255,255,255,0.10)",
          background: "rgba(255,255,255,0.04)",
          borderRadius: 18,
          padding: 14,
        }}
      >
        <div style={{ fontWeight: 700 }}>Example cards (mock)</div>
        <ul style={{ marginTop: 10, opacity: 0.8 }}>
          <li>Balcony camera: offline</li>
          <li>Kitchen stove sensor: unknown</li>
          <li>Air quality: unknown</li>
        </ul>
      </div>
    </div>
  );
}

// ✅ both exports to avoid "no default export" errors
export default Devices;
export { Devices };
