# NukaMotion Architecture

Motion + Agent Coordination Design

---

## Overview

NukaMotion consists of **two tightly-coupled subsystems**:

1. **Motion Perception System** (MoveNet-based squat detection)
2. **Agent & Alarm System** (Nuka Agent powered by Qwen LLM)

They communicate through a **local HTTP interface**, allowing real-time motion
events to control alarm state deterministically.

---

## High-Level Components

```
+-------------------+        HTTP (POST /rep)
|                   |  -------------------->
|   Motion Engine   |                        |
|  (MoveNet + CV)   |                        |
|                   |  <--------------------
+-------------------+        HTTP (GET /status)
           |
           |  Squat Detected
           v
+-------------------+
| Squat State Machine|
|  STAND/DOWN/COUNT |
+-------------------+
```

```
+----------------------------+
|        Nuka Agent          |
|----------------------------|
| - Alarm Scheduler          |
| - SQLite Persistence       |
| - Session State Machine    |
| - LLM Intent Parser        |
| - Audio Control Logic      |
+----------------------------+
```

---

## Alarm Lifecycle

```
IDLE
 |
 | (Alarm Time Reached)
 v
RINGING  <-- alarm sound looping
 |
 | (Human motion detected)
 v
SQUAT_ACTIVE  <-- alarm muted, counting reps
 |
 | (Reps completed)
 v
UNLOCKED
 |
 | (Stop confirmed)
 v
DONE
```

Key rule:
- **Alarm sound resumes automatically** if motion stops before completion.

---

## Motion → Agent Timing Flow

```
Time ─────────────────────────────────────────────>

[ Alarm rings ]
      🔊🔊🔊🔊🔊

User starts squatting
      |
      v
MoveNet detects DOWN → STAND
      |
      v
POST /rep
      |
      v
NukaCore.report_rep()
      |
      v
rep_count += 1
      |
      v
🔔 "ding" sound

(repeat)

If motion stops > timeout:
      |
      v
Alarm resumes 🔊

If reps == target:
      |
      v
Alarm permanently stops
      |
      v
LLM encouragement (once)
```

---

## Motion Engine Details

**Input**
- USB Camera (side view)

**Pipeline**
```
Frame
 → Letterbox Resize
   → MoveNet Inference
     → Keypoints
       → Knee Angle
         → EMA Smoothing
           → Squat State Machine
```

**Output**
- `SQUAT_DONE` event
- Triggers HTTP POST `/rep`

---

## Agent Responsibilities

### Deterministic (No LLM)
- Alarm scheduling
- Squat counting
- Alarm enforcement
- Audio timing
- Timeout handling
- Persistence (SQLite)

### LLM-Assisted (Qwen via Ollama)
Used **only where ambiguity exists**:

| Task | LLM |
|----|----|
| Parse natural language commands | ✅ |
| Convert intent → JSON | ✅ |
| Generate encouragement text | ✅ |
| Alarm timing logic | ❌ |
| Squat counting | ❌ |

This guarantees **real-time safety** and **low latency**.

---

## Why This Architecture Works

- Motion system runs at camera frame rate
- Agent logic remains deterministic
- LLM is never in the critical path
- Clear separation of perception vs decision
- Easy to extend with new motion types

---

## Future Extensions

- GPU/TensorRT inference
- Multiple motion types (jump, plank)
- BLE / wearable integration
- Smart speaker / TTS output
- Cloud-free offline mode

---

**Motion proves wakefulness.**
