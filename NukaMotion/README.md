# NukaMotion

**Motion-based wake-up system.  
No buttons. No cheating. Just move.**

> *Passwords prove identity. Motion proves wakefulness.*

---

## 🧠 What is NukaMotion?

**NukaMotion** is an experimental wake-up system that requires **physical movement**
(e.g. squats) to unlock an alarm.

Instead of tapping a button half-asleep, you must **prove that you are awake** —
with your body.

The system combines:

- **Motion intelligence** (real-time pose & squat detection)
- **Agent intelligence** (LLM-powered alarm control & interaction)

to create a wake-up experience that is both **physically enforced** and **cognitively aware**.

---

## 🚀 Why NukaMotion?

Traditional alarms are easy to silence and easy to regret.

NukaMotion is built on a simple idea:

> **If your body is moving, you are awake.**

By enforcing motion as a wake-up condition, NukaMotion:
- Prevents unconscious snoozing
- Forces blood flow and muscle activation
- Creates a short but effective *wake-up ritual*
- Eliminates “cheating” via half-awake button presses

---

## 🧩 What Makes It Different?

NukaMotion is **not just a vision demo**.

It is a **motion-gated intelligent agent system**:

| Layer | Responsibility |
|-----|----------------|
| **Motion Layer** | Detects real squats using pose estimation |
| **Agent Layer** | Manages alarms, rules, states, and user intent |
| **Language Layer** | Understands natural language commands via LLM |

Motion unlocks the alarm.  
Language controls the system.

---

## 🎥 Demo

NukaMotion supports a debug/demo mode with:
- Real-time skeleton overlay
- Knee angle visualization
- Squat count and state
- Optional recorded output video (`debug_out.mp4`)

Designed to be **demo-friendly** and **presentation-ready**.

---

## 🏗 System Overview

### Hardware
- NVIDIA Jetson (Nano / Orin, CPU-first, GPU planned)
- USB camera (side-view recommended)
- Speaker (USB / I2S planned for alarm & feedback)

### Software
- MoveNet (SinglePose, Lightning)
- TFLite Runtime (CPU inference)
- OpenCV + GStreamer
- Python-based squat state machine
- SQLite (alarm & session persistence)
- Ollama + Qwen LLM (agent intelligence)

---

## 🧠 Agent Architecture (Nuka Agent)

NukaMotion includes a local AI agent (**Nuka Agent**) powered by **Qwen LLM** via **Ollama**.

### What the Agent Does

The agent is responsible for:

- Understanding **natural language commands**
- Managing **alarm lifecycle**
- Enforcing **motion-based unlock rules**
- Coordinating with the motion system via HTTP

### Example Interactions

```
Set an alarm tomorrow morning at seven
Cancel my morning alarm
Change tomorrow 6 to 8
List all alarms
Close all alarms
```

---

## 🔄 Motion ↔ Agent Integration

The system is intentionally **decoupled**:

### MoveNet Motion Process
- Runs continuous pose estimation
- Detects valid squat events
- On each valid squat:
  POST http://127.0.0.1:8008/rep

### Nuka Agent HTTP API
```
POST /rep     → count one squat
POST /stop    → attempt to stop alarm
GET  /status  → query active session
```

---

## 🏃 How Squats Are Detected

- Side-view human pose is tracked using MoveNet
- Knee joint angle is calculated from hip–knee–ankle keypoints
- State machine detects STAND → DOWN → STAND
- Each valid cycle counts as **one squat**
- Alarm unlocks only after **N valid squats**

---

## 🔔 Alarm Logic (No Cheating)

- Alarm rings continuously until motion begins
- During squats: alarm pauses, short ding per rep
- If user stops moving too long: alarm resumes
- Only after completing all reps: alarm can be stopped

---

## ⚠️ Current Limitations

- Single-person only
- Side-view works best
- CPU inference (~10–13 FPS)
- No speech I/O yet
- Experimental prototype

---

## 🔮 Future Work

- TensorRT acceleration
- Audio & voice integration
- Speech-to-text / text-to-speech
- More motion types
- Embedded deployment

---

## 🧪 Project Status

Part of **Nuka Lab** — focused on:
- Human–machine interaction
- Edge AI
- LLM + physical world integration

---

## 🧾 License

MIT License

---

**Wake up.  
Move.  
Unlock.**
