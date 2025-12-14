#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import re
from datetime import timedelta

from dateparser.conf import SettingValidationError

import dateparser
import requests

TZ_NAME = "Europe/Berlin"
TZ = ZoneInfo(TZ_NAME)

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

DB_PATH = "nuka.db"


# -------------------------
# Persistence (SQLite)
# -------------------------
def db_init():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alarms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time_iso TEXT NOT NULL,
        label TEXT,
        target_reps INTEGER NOT NULL DEFAULT 10,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alarm_id INTEGER NOT NULL,
        state TEXT NOT NULL,                -- RINGING, SQUAT_ACTIVE, UNLOCKED, DONE
        current_reps INTEGER NOT NULL DEFAULT 0,
        target_reps INTEGER NOT NULL DEFAULT 10,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        FOREIGN KEY(alarm_id) REFERENCES alarms(id)
    )
    """)
    conn.commit()
    conn.close()


def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def now_local() -> datetime:
    return datetime.now(tz=TZ)


def iso(dt: datetime) -> str:
    # Always keep timezone-aware ISO 8601
    return dt.astimezone(TZ).isoformat(timespec="seconds")





def parse_time_text(time_text: str, base: datetime | None = None) -> datetime | None:
    """
    Robust time parser:
      - Never crashes.
      - Tries dateparser first (with safe settings).
      - Falls back to regex for relative times (CN/EN).
    """
    if base is None:
        base = now_local()

    text = (time_text or "").strip()
    if not text:
        return None

    low = text.lower()

    # ---------- Layer 1: dateparser (safe settings) ----------
    # IMPORTANT: do NOT use unsupported settings.
    settings = {
        "TIMEZONE": TZ_NAME,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base,
    }

    dt = None
    try:
        # Hint languages to improve CN/EN parsing; supported across many versions
        dt = dateparser.parse(text, settings=settings, languages=["en", "zh"])
    except SettingValidationError:
        # If some setting key is not supported in this dateparser version, retry with minimal settings
        try:
            dt = dateparser.parse(text, settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": base,
                "RETURN_AS_TIMEZONE_AWARE": True,
            }, languages=["en", "zh"])
        except Exception:
            dt = None
    except Exception:
        dt = None

    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        return dt.astimezone(TZ)

    # ---------- Layer 2: regex fallback (relative time, EN) ----------
    # "in 1 minute", "in one minute", "1 minute later"
    # digits
    m = re.search(r'\b(in\s*)?(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b', low)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        if unit.startswith(("second", "sec", "s")):
            return base + timedelta(seconds=n)
        if unit.startswith(("minute", "min", "m")):
            return base + timedelta(minutes=n)
        if unit.startswith(("hour", "hr", "h")):
            return base + timedelta(hours=n)
        if unit.startswith(("day", "d")):
            return base + timedelta(days=n)

    # "one minute", "two minutes"
    word_to_int = {
        "one": 1, "a": 1, "an": 1,
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    m = re.search(r'\b(in\s*)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s*(minute|hour|day)s?\b', low)
    if m:
        n = word_to_int.get(m.group(2), None)
        unit = m.group(3)
        if n is not None:
            if unit == "minute":
                return base + timedelta(minutes=n)
            if unit == "hour":
                return base + timedelta(hours=n)
            if unit == "day":
                return base + timedelta(days=n)

    # ---------- Layer 3: regex fallback (relative time, CN) ----------
    # "1分钟后", "2小时后", "10秒后", "3天后"
    m = re.search(r'(\d+)\s*(秒|分钟|分|小时|天)\s*后', text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "秒":
            return base + timedelta(seconds=n)
        if unit in ("分钟", "分"):
            return base + timedelta(minutes=n)
        if unit == "小时":
            return base + timedelta(hours=n)
        if unit == "天":
            return base + timedelta(days=n)

    return None



# -------------------------
# Ollama client helpers
# -------------------------
def ollama_chat(messages, tools=None, temperature=0.2):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    # Ollama supports tools; we keep MVP with structured JSON in content for robustness
    if tools is not None:
        payload["tools"] = tools

    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def llm_intent(user_text: str) -> dict:
    """
    Phase A: return ONLY a JSON object describing intent.
    We validate/repair with a couple retries.
    """
    system = (
        "You are Nuka, a strict command parser for an alarm + squat system.\n"
        "Return ONLY a single-line JSON object. No markdown, no extra text.\n"
        "Timezone is Europe/Berlin.\n"
        "\n"
        "Allowed intents:\n"
        "CREATE_ALARM: set a new alarm\n"
        "CANCEL_ALARM: cancel an existing alarm\n"
        "UPDATE_ALARM: change an existing alarm time\n"
        "LIST_ALARMS: list alarms\n"
        "STOP_ALARM: user tries to stop the currently ringing alarm\n"
        "HELP: user asks how to use\n"
        "CHAT: everything else\n"
        "\n"
        "JSON schemas:\n"
        "CREATE_ALARM: {\"intent\":\"CREATE_ALARM\",\"time_text\":\"...\",\"target_reps\":10,\"label\":\"optional\"}\n"
        "CANCEL_ALARM: {\"intent\":\"CANCEL_ALARM\",\"which\":\"...\"}\n"
        "UPDATE_ALARM: {\"intent\":\"UPDATE_ALARM\",\"from\":\"...\",\"to\":\"...\"}\n"
        "LIST_ALARMS: {\"intent\":\"LIST_ALARMS\"}\n"
        "STOP_ALARM: {\"intent\":\"STOP_ALARM\"}\n"
        "HELP: {\"intent\":\"HELP\"}\n"
        "CHAT: {\"intent\":\"CHAT\",\"text\":\"...\"}\n"
        "\n"
        "Rules:\n"
        "- If user says 'tomorrow morning 7', put that in time_text.\n"
        "- If user says cancel 'tomorrow morning alarm', put that in which.\n"
        "- If user says change 6 to 8, put from='tomorrow 6' to='tomorrow 8' if implied.\n"
        "- target_reps defaults to 10 if not specified.\n"
    )

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]

    last_err = None
    for _ in range(3):
        out = ollama_chat(msgs, temperature=0.0)
        content = out["message"]["content"].strip()

        # Try strict JSON
        try:
            obj = json.loads(content)
            if not isinstance(obj, dict) or "intent" not in obj:
                raise ValueError("Missing intent")
            return obj
        except Exception as e:
            last_err = str(e)
            # Ask the model to repair to valid JSON only
            msgs.append({"role": "assistant", "content": content})
            msgs.append({
                "role": "user",
                "content": f"Fix: return ONLY valid JSON object. Error: {last_err}"
            })

    return {"intent": "CHAT", "text": f"(parse failed) {user_text}"}


def llm_say(style: str, context: dict) -> str:
    """
    Phase B: generate user-facing text. style in: ring, encourage, refuse_stop, confirm
    """
    system = (
        "You are Nuka, a friendly fitness alarm assistant. "
        "Be concise, energetic, and positive. 1-2 short sentences."
    )
    user = {"style": style, "context": context}
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
    out = ollama_chat(msgs, temperature=0.7)
    return out["message"]["content"].strip()


# -------------------------
# Runtime session state
# -------------------------
@dataclass
class ActiveSession:
    alarm_id: int
    session_id: int
    state: str            # RINGING/SQUAT_ACTIVE/UNLOCKED
    current_reps: int
    target_reps: int
    started_at: datetime


class NukaCore:
    def __init__(self):
        self.conn = db_conn()
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.active: ActiveSession | None = None

        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)

    # ---- Alarm CRUD ----
    def create_alarm(self, time_dt: datetime, target_reps: int = 10, label: str | None = None) -> int:
        with self.lock:
            cur = self.conn.cursor()
            t = iso(time_dt)
            now = iso(now_local())
            cur.execute(
                "INSERT INTO alarms (time_iso,label,target_reps,enabled,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (t, label, int(target_reps), 1, now, now),
            )
            self.conn.commit()
            return cur.lastrowid

    def list_alarms(self) -> list[dict]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id,time_iso,label,target_reps,enabled FROM alarms ORDER BY time_iso ASC")
            rows = cur.fetchall()
        out = []
        for (aid, t, label, reps, enabled) in rows:
            out.append({"id": aid, "time_iso": t, "label": label, "target_reps": reps, "enabled": bool(enabled)})
        return out

    def find_alarm_by_text(self, which: str) -> dict | None:
        """
        MVP heuristic:
        - if which contains an integer -> treat as alarm_id
        - else parse time_text and match nearest alarm within 2 hours
        - else match label substring
        """
        which = (which or "").strip()
        alarms = self.list_alarms()
        if not alarms:
            return None

        # id match
        if which.isdigit():
            aid = int(which)
            for a in alarms:
                if a["id"] == aid:
                    return a

        # label match
        for a in alarms:
            if a["label"] and which and which.lower() in a["label"].lower():
                return a

        # time match
        dt = parse_time_text(which)
        if dt:
            target = dt
            best = None
            best_abs = None
            for a in alarms:
                if not a["enabled"]:
                    continue
                try:
                    at = datetime.fromisoformat(a["time_iso"])
                except Exception:
                    continue
                diff = abs((at - target).total_seconds())
                if best is None or diff < best_abs:
                    best = a
                    best_abs = diff
            if best and best_abs is not None and best_abs <= 2 * 3600:
                return best

        # fallback: nearest future enabled alarm
        nowt = now_local()
        future = []
        for a in alarms:
            if not a["enabled"]:
                continue
            try:
                at = datetime.fromisoformat(a["time_iso"])
            except Exception:
                continue
            if at >= nowt:
                future.append((at, a))
        if future:
            future.sort(key=lambda x: x[0])
            return future[0][1]
        return None

    def cancel_alarm(self, alarm_id: int) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("UPDATE alarms SET enabled=0, updated_at=? WHERE id=?", (iso(now_local()), alarm_id))
            self.conn.commit()
            return cur.rowcount > 0

    def update_alarm(self, alarm_id: int, new_time: datetime) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("UPDATE alarms SET time_iso=?, updated_at=? WHERE id=?",
                        (iso(new_time), iso(now_local()), alarm_id))
            self.conn.commit()
            return cur.rowcount > 0

    # ---- Session / ring / reps ----
    def _start_session_for_alarm(self, alarm: dict):
        # Create session record
        with self.lock:
            cur = self.conn.cursor()
            started = now_local()
            cur.execute(
                "INSERT INTO sessions (alarm_id,state,current_reps,target_reps,started_at) VALUES (?,?,?,?,?)",
                (alarm["id"], "RINGING", 0, alarm["target_reps"], iso(started)),
            )
            self.conn.commit()
            sid = cur.lastrowid
            self.active = ActiveSession(
                alarm_id=alarm["id"],
                session_id=sid,
                state="RINGING",
                current_reps=0,
                target_reps=alarm["target_reps"],
                started_at=started,
            )

        # user-facing ring message
        msg = llm_say("ring", {
            "alarm_time": alarm["time_iso"],
            "target_reps": alarm["target_reps"],
            "label": alarm["label"]
        })
        print(f"\n⏰ ALARM! {msg}")
        print(f"👉 Start squats now. Type 'rep' for each squat. Target: {alarm['target_reps']}\n")
        # move to active
        with self.lock:
            if self.active:
                self.active.state = "SQUAT_ACTIVE"
                cur = self.conn.cursor()
                cur.execute("UPDATE sessions SET state=? WHERE id=?", ("SQUAT_ACTIVE", self.active.session_id))
                self.conn.commit()

    def report_rep(self):
        with self.lock:
            if not self.active or self.active.state not in ("SQUAT_ACTIVE", "UNLOCKED"):
                print("No active squat session. (Wait for alarm or create one.)")
                return
            self.active.current_reps += 1
            cur_reps = self.active.current_reps
            target = self.active.target_reps

            # persist
            cur = self.conn.cursor()
            cur.execute("UPDATE sessions SET current_reps=? WHERE id=?", (cur_reps, self.active.session_id))
            self.conn.commit()

        # "beep"
        print("🔔 ding")

        if cur_reps >= target:
            with self.lock:
                if self.active:
                    self.active.state = "UNLOCKED"
                    cur = self.conn.cursor()
                    cur.execute("UPDATE sessions SET state=? WHERE id=?", ("UNLOCKED", self.active.session_id))
                    self.conn.commit()
            msg = llm_say("encourage", {"current": cur_reps, "target": target, "done": True})
            print(f"✅ {cur_reps}/{target} {msg}")
            print("👉 You can now type 'stop' to stop the alarm.\n")
        else:
            msg = llm_say("encourage", {"current": cur_reps, "target": target, "done": False})
            print(f"💪 {cur_reps}/{target} {msg}")

    def try_stop_alarm(self):
        with self.lock:
            if not self.active:
                print("No alarm is ringing right now.")
                return
            state = self.active.state
            cur = self.active.current_reps
            target = self.active.target_reps
            sid = self.active.session_id

        if state != "UNLOCKED":
            msg = llm_say("refuse_stop", {"current": cur, "target": target})
            print(f"❌ Can't stop yet. {msg} (remaining: {max(0, target-cur)})")
            return

        # stop and finalize
        with self.lock:
            cur2 = self.conn.cursor()
            cur2.execute("UPDATE sessions SET state=?, ended_at=? WHERE id=?",
                        ("DONE", iso(now_local()), sid))
            self.conn.commit()
            self.active = None

        msg = llm_say("confirm", {"result": "stopped", "target": target})
        print(f"🛑 Alarm stopped. {msg}\n")

    # ---- scheduler loop ----
    def _scheduler_loop(self):
        while not self.stop_flag.is_set():
            # don't start new if active
            with self.lock:
                active_exists = self.active is not None
            if not active_exists:
                due_alarm = self._get_next_due_alarm()
                if due_alarm:
                    self._start_session_for_alarm(due_alarm)
            time.sleep(0.5)

    def _get_next_due_alarm(self) -> dict | None:
        alarms = self.list_alarms()
        if not alarms:
            return None
        nowt = now_local()
        # find enabled alarm whose time <= now and not too old (grace window)
        grace = timedelta(seconds=10)
        due = []
        for a in alarms:
            if not a["enabled"]:
                continue
            try:
                at = datetime.fromisoformat(a["time_iso"])
            except Exception:
                continue
            if at <= nowt <= (at + grace):
                due.append((at, a))
        if due:
            due.sort(key=lambda x: x[0])
            return due[0][1]
        return None

    def start(self):
        self.scheduler_thread.start()

    def shutdown(self):
        self.stop_flag.set()
        time.sleep(0.2)
        try:
            self.conn.close()
        except Exception:
            pass


# -------------------------
# CLI
# -------------------------
HELP_TEXT = """
Commands:
  - natural language:
      "set an alarm tomorrow 7am"
      "cancel tomorrow morning alarm"
      "change tomorrow 6 to 8"
      "list alarms"
  - special:
      rep      -> report one squat rep (for MVP)
      stop     -> try stop current alarm (only after reps complete)
      exit     -> quit
"""

def render_alarms(alarms: list[dict]):
    if not alarms:
        return "No alarms."
    lines = ["Alarms:"]
    for a in alarms:
        en = "ON" if a["enabled"] else "OFF"
        label = f" ({a['label']})" if a["label"] else ""
        lines.append(f"  #{a['id']}  {a['time_iso']}  reps={a['target_reps']}  {en}{label}")
    return "\n".join(lines)


def main():
    db_init()
    core = NukaCore()
    core.start()

    print("Nuka MVP is running. Type /help for help.")
    try:
        while True:
            s = input(">>> ").strip()
            if not s:
                continue
            if s in ("/help", "help", "?"):
                print(HELP_TEXT)
                continue
            if s in ("exit", "quit"):
                break
            if s == "rep":
                core.report_rep()
                continue
            if s == "stop":
                core.try_stop_alarm()
                continue

            # Phase A: intent
            intent = llm_intent(s)
            it = intent.get("intent", "CHAT")

            if it == "HELP":
                print(HELP_TEXT)
                continue

            if it == "LIST_ALARMS":
                alarms = core.list_alarms()
                print(render_alarms(alarms))
                continue

            if it == "CREATE_ALARM":
                time_text = intent.get("time_text", "")
                reps = int(intent.get("target_reps", 10) or 10)
                label = intent.get("label", None)
                dt = parse_time_text(time_text)
                if not dt:
                    print(f"❌ Couldn't parse time: {time_text}")
                    continue
                aid = core.create_alarm(dt, target_reps=reps, label=label)
                msg = llm_say("confirm", {"result": "created", "alarm_id": aid, "time": iso(dt), "target_reps": reps})
                print(f"✅ Created alarm #{aid} at {iso(dt)} (reps={reps}). {msg}")
                continue

            if it == "CANCEL_ALARM":
                which = intent.get("which", "")
                a = core.find_alarm_by_text(which)
                if not a:
                    print("❌ No matching alarm found.")
                    continue
                ok = core.cancel_alarm(a["id"])
                msg = llm_say("confirm", {"result": "canceled", "alarm_id": a["id"], "time": a["time_iso"]})
                print(f"✅ Canceled alarm #{a['id']} ({a['time_iso']}). {msg}" if ok else "❌ Cancel failed.")
                continue

            if it == "UPDATE_ALARM":
                frm = intent.get("from", "")
                to = intent.get("to", "")
                a = core.find_alarm_by_text(frm)
                if not a:
                    print("❌ No matching alarm to update.")
                    continue
                new_dt = parse_time_text(to)
                if not new_dt:
                    print(f"❌ Couldn't parse new time: {to}")
                    continue
                ok = core.update_alarm(a["id"], new_dt)
                msg = llm_say("confirm", {"result": "updated", "alarm_id": a["id"], "old": a["time_iso"], "new": iso(new_dt)})
                print(f"✅ Updated alarm #{a['id']} -> {iso(new_dt)}. {msg}" if ok else "❌ Update failed.")
                continue

            if it == "STOP_ALARM":
                core.try_stop_alarm()
                continue

            # fallback chat
            text = intent.get("text", s)
            reply = llm_say("confirm", {"result": "chat", "text": text})
            print(reply)

    except KeyboardInterrupt:
        pass
    finally:
        core.shutdown()
        print("Bye.")


if __name__ == "__main__":
    main()
