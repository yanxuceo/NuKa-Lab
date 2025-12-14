#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sqlite3
import threading
import time
import re
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import dateparser
from dateparser.conf import SettingValidationError
import requests


from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


TZ_NAME = "Europe/Berlin"
TZ = ZoneInfo(TZ_NAME)

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

DB_PATH = "nuka.db"

# -------------------------
# Encouragement optimization
# -------------------------
# Milestones that will trigger an LLM call for encouragement (fast most of the time)
DEFAULT_MILESTONES = {1, 3, 5, 8, 9}  # plus target reps
# Local fast phrases (no LLM, super quick)
ENCOURAGE_POOL_EN = [
    "Nice!",
    "Good rep!",
    "Keep going!",
    "Strong squat!",
    "Great form!",
    "Solid!",
    "You got this!",
    "Stay steady!",
    "Keep the rhythm!",
    "Power up!",
    "Strong legs!",
    "Clean rep!",
    "Let's go!",
    "Awesome!",
    "Almost there!",
    "Breathe and squat!",
    "Great work!",
    "Focus!",
    "Smooth!",
    "One rep at a time!",
]
ENCOURAGE_POOL_ZH = [
    "漂亮！",
    "好！",
    "继续！",
    "很稳！",
    "动作不错！",
    "节奏很好！",
    "加油！",
    "稳住！",
    "很强！",
    "就这样！",
    "腿很给力！",
    "再来一个！",
    "冲！",
    "不错！",
    "快到了！",
    "呼吸稳住！",
    "干得漂亮！",
    "专注！",
    "很顺！",
    "一步一步来！",
]


def detect_lang(text: str) -> str:
    # very simple CJK detection
    if any("\u4e00" <= ch <= "\u9fff" for ch in (text or "")):
        return "zh"
    return "en"


def pick_local_phrase(lang: str, remaining: int, done: bool) -> str:
    if done:
        return "完成！" if lang == "zh" else "Done!"
    pool = ENCOURAGE_POOL_ZH if lang == "zh" else ENCOURAGE_POOL_EN
    # add a tiny hint when close
    if remaining == 1:
        return "最后一个！" if lang == "zh" else "One more!"
    if remaining == 2:
        return "还差两个！" if lang == "zh" else "Two left!"
    return random.choice(pool)


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
    return dt.astimezone(TZ).isoformat(timespec="seconds")


# -------------------------
# Time parsing (robust)
# -------------------------
def parse_time_text(time_text: str, base: datetime | None = None) -> datetime | None:
    """
    Robust time parser:
      - Never crashes.
      - HARD-RULES for common ambiguous cases (tomorrow morning X).
      - Tries dateparser first (safe settings).
      - Falls back to regex for relative times (CN/EN).
    """
    if base is None:
        base = now_local()

    text = (time_text or "").strip()
    if not text:
        return None

    low = text.lower()

    # ==========================================================
    # HARD RULE 1: "tomorrow morning X" / "明天早上X点" / "morgen früh X"
    # Fix dateparser ambiguity for "tomorrow morning 7 (o'clock)" etc.
    # ==========================================================
    # Accept:
    #  - tomorrow morning 7 / tomorrow morning seven / tomorrow morning 7 o'clock
    #  - 明天早上7 / 明天早上七点
    #  - morgen früh 7 / morgen morgen 7 Uhr (loosely)
    #
    # Notes:
    #  - If user says "morning/早上/früh" and gives an hour without am/pm,
    #    we force it into 05:00–11:00 and set it to TOMORROW at HH:00.
    #
    #  - This is deterministic and avoids "+24h/+48h" surprises.
    # ==========================================================
    word_to_int = {
        "one": 1, "a": 1, "an": 1,
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12,
    }
    zh_to_int = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十一": 11, "十二": 12,
    }

    # English / German style: tomorrow/morgen + morning/früh/morgen + hour
    m = re.search(
        r'\b(tomorrow|morgen|明天)\b.*\b(morning|früh|frueh|morgen|早上|上午)\b.*\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b',
        low
    )
    if m:
        h_raw = m.group(3)
        if h_raw.isdigit():
            hour = int(h_raw)
        else:
            hour = word_to_int.get(h_raw, None)

        if hour is not None:
            # morning heuristic clamp
            hour = max(5, min(hour, 11))
            day = (base + timedelta(days=1)).date()
            return datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=TZ)

    # Chinese style: 明天早上七点 / 明天上午7点 / 明天早上7
    m = re.search(r'(明天).*(早上|上午).*(\d{1,2}|[一二两三四五六七八九十]{1,3})\s*(点|點)?', text)
    if m:
        h_raw = m.group(3)
        if h_raw.isdigit():
            hour = int(h_raw)
        else:
            hour = zh_to_int.get(h_raw, None)

        if hour is not None:
            hour = max(5, min(hour, 11))
            day = (base + timedelta(days=1)).date()
            return datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=TZ)

    # ==========================================================
    # Layer 1: dateparser (safe settings)
    # ==========================================================
    settings = {
        "TIMEZONE": TZ_NAME,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base,
    }

    dt = None
    try:
        dt = dateparser.parse(text, settings=settings, languages=["en", "zh", "de"])
    except SettingValidationError:
        try:
            dt = dateparser.parse(
                text,
                settings={
                    "PREFER_DATES_FROM": "future",
                    "RELATIVE_BASE": base,
                    "RETURN_AS_TIMEZONE_AWARE": True,
                },
                languages=["en", "zh", "de"],
            )
        except Exception:
            dt = None
    except Exception:
        dt = None

    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        return dt.astimezone(TZ)

    # ==========================================================
    # Layer 2: regex fallback (relative EN digits)
    # e.g. "in 2 minutes", "2 minutes", "2 min", "in 10 sec"
    # ==========================================================
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

    # ==========================================================
    # Layer 2b: regex fallback (relative EN words)
    # ==========================================================
    word_to_int_small = {
        "one": 1, "a": 1, "an": 1,
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    m = re.search(r'\b(in\s*)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s*(minute|hour|day)s?\b', low)
    if m:
        n = word_to_int_small.get(m.group(2), None)
        unit = m.group(3)
        if n is not None:
            if unit == "minute":
                return base + timedelta(minutes=n)
            if unit == "hour":
                return base + timedelta(hours=n)
            if unit == "day":
                return base + timedelta(days=n)

    # ==========================================================
    # Layer 3: regex fallback (relative CN)
    # e.g. "1分钟后", "2小时后", "10秒后", "3天后"
    # ==========================================================
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
    if tools is not None:
        payload["tools"] = tools

    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def llm_intent(user_text: str) -> dict:
    """
    Phase A: return ONLY a JSON object describing intent.
    Keep it fast and deterministic: temperature=0.
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
        "- Keep time_text/from/to/which in the user's original language. Do NOT translate.\n"
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
        try:
            obj = json.loads(content)
            if not isinstance(obj, dict) or "intent" not in obj:
                raise ValueError("Missing intent")
            return obj
        except Exception as e:
            last_err = str(e)
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": f"Fix: return ONLY valid JSON object. Error: {last_err}"})

    return {"intent": "CHAT", "text": f"(parse failed) {user_text}"}


def llm_say(style: str, context: dict) -> str:
    """
    Phase B: generate user-facing text for ring/encourage/refuse_stop.
    IMPORTANT:
      - Only talk about squats + rep counting
      - Never mention time-of-day/dates or other exercises
      - Keep it short
    """
    system = (
        "You are Nuka, a squat alarm coach.\n"
        "Rules:\n"
        "- ONLY talk about SQUATS and REP counting.\n"
        "- NEVER mention steps, push-ups, running, days, weeks, calendars.\n"
        "- NEVER mention specific time-of-day or dates.\n"
        "- Use ONLY numbers provided (current_reps/target_reps/remaining). Do not invent numbers.\n"
        "- Output 1 short sentence, ideally <= 10 words.\n"
        "- Language: if lang is 'zh' use Chinese, else use English.\n"
    )

    safe_ctx = {
        "style": style,
        "exercise": "squat",
        "lang": context.get("lang", "en"),
        "current_reps": context.get("current_reps"),
        "target_reps": context.get("target_reps"),
        "remaining": context.get("remaining"),
        "done": context.get("done"),
    }

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(safe_ctx, ensure_ascii=False)},
    ]

    out = ollama_chat(msgs, temperature=0.35)
    return out["message"]["content"].strip()


@dataclass
class ActiveSession:
    alarm_id: int
    session_id: int
    state: str            # RINGING/SQUAT_ACTIVE/UNLOCKED
    current_reps: int
    target_reps: int
    started_at: datetime
    lang: str = "en"


class NukaCore:
    def __init__(self):
        self.conn = db_conn()
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.default_lang = "en"
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
        which = (which or "").strip()
        alarms = self.list_alarms()
        if not alarms:
            return None

        if which.isdigit():
            aid = int(which)
            for a in alarms:
                if a["id"] == aid:
                    return a

        for a in alarms:
            if a["label"] and which and which.lower() in a["label"].lower():
                return a

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
        with self.lock:
            cur = self.conn.cursor()
            started = now_local()
            cur.execute(
                "INSERT INTO sessions (alarm_id,state,current_reps,target_reps,started_at) VALUES (?,?,?,?,?)",
                (alarm["id"], "RINGING", 0, alarm["target_reps"], iso(started)),
            )
            self.conn.commit()
            sid = cur.lastrowid

            # default language for ring/encourage can be English; you can make it smarter later
            self.active = ActiveSession(
                alarm_id=alarm["id"],
                session_id=sid,
                state="RINGING",
                current_reps=0,
                target_reps=alarm["target_reps"],
                started_at=started,
                lang=self.default_lang,

            )

        # ring message (no time fed to model)
        lang = self.active.lang if self.active else "en"
        msg = llm_say("ring", {
            "current_reps": 0,
            "target_reps": alarm["target_reps"],
            "remaining": alarm["target_reps"],
            "done": False,
            "lang": lang
        })
        print(f"\n⏰ ALARM! {msg}")
        print(f"👉 Start squats now. Type 'rep' for each squat. Target: {alarm['target_reps']}\n")

        with self.lock:
            if self.active:
                self.active.state = "SQUAT_ACTIVE"
                cur = self.conn.cursor()
                cur.execute("UPDATE sessions SET state=? WHERE id=?", ("SQUAT_ACTIVE", self.active.session_id))
                self.conn.commit()

    def _encourage(self, cur_reps: int, target: int, done: bool, lang: str) -> str:
        remaining = max(0, target - cur_reps)
        # milestones trigger LLM (more “alive”), others use local fast phrases
        milestones = set(DEFAULT_MILESTONES)
        milestones.add(target)

        if done or (cur_reps in milestones):
            return llm_say("encourage", {
                "current_reps": cur_reps,
                "target_reps": target,
                "remaining": remaining,
                "done": done,
                "lang": lang
            })

        return pick_local_phrase(lang=lang, remaining=remaining, done=done)


    def report_rep(self):
        with self.lock:
            if not self.active or self.active.state not in ("SQUAT_ACTIVE", "UNLOCKED"):
                print("No active squat session. (Wait for alarm or create one.)")
                return
            self.active.current_reps += 1
            cur_reps = self.active.current_reps
            target = self.active.target_reps
            lang = self.active.lang

            # persist
            cur = self.conn.cursor()
            cur.execute("UPDATE sessions SET current_reps=? WHERE id=?", (cur_reps, self.active.session_id))
            self.conn.commit()

        # per-rep feedback: only beep (fast)
        print("🔔 ding")

        # when done, unlock + speak ONE sentence (LLM once)
        if cur_reps >= target:
            with self.lock:
                if self.active:
                    self.active.state = "UNLOCKED"
                    cur = self.conn.cursor()
                    cur.execute("UPDATE sessions SET state=? WHERE id=?", ("UNLOCKED", self.active.session_id))
                    self.conn.commit()

            # ONE LLM call only here
            msg = llm_say("encourage", {
                "current_reps": cur_reps,
                "target_reps": target,
                "remaining": 0,
                "done": True,
                "lang": lang
            })
            print(f"✅ {cur_reps}/{target} {msg}")
            print("👉 You can now type 'stop' to stop the alarm.\n")


    def try_stop_alarm(self):
        with self.lock:
            if not self.active:
                print("No alarm is ringing right now.")
                return
            state = self.active.state
            cur = self.active.current_reps
            target = self.active.target_reps
            sid = self.active.session_id
            lang = self.active.lang

        if state != "UNLOCKED":
            remaining = max(0, target - cur)
            if lang == "zh":
                print(f"❌ 还不能关！还差 {remaining} 个深蹲。")
            else:
                print(f"❌ Not yet! {remaining} squats remaining.")
            return

        with self.lock:
            cur2 = self.conn.cursor()
            cur2.execute("UPDATE sessions SET state=?, ended_at=? WHERE id=?",
                        ("DONE", iso(now_local()), sid))
            self.conn.commit()
            self.active = None

        # stop confirmation: NO LLM (fast + deterministic)
        if lang == "zh":
            print(f"🛑 闹钟已关闭。太棒了！你完成了 {target} 个深蹲！\n")
        else:
            print(f"🛑 Alarm stopped. Nice work — {target} squats done!\n")

    # ---- scheduler loop ----
    def _scheduler_loop(self):
        while not self.stop_flag.is_set():
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


def start_http_api(core, host="127.0.0.1", port=8008):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code=200, body="OK", content_type="text/plain; charset=utf-8"):
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                try:
                    self.wfile.write(body.encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError):
                    pass
            except Exception:
                # don’t crash handler thread for any response errors
                pass


        def do_GET(self):
            p = urlparse(self.path).path
            if p != "/status":
                return self._send(404, "Not Found")

            with core.lock:
                a = core.active
                if not a:
                    payload = {"active": False}
                else:
                    payload = {
                        "active": True,
                        "state": a.state,
                        "current_reps": a.current_reps,
                        "target_reps": a.target_reps,
                        "alarm_id": a.alarm_id,
                        "session_id": a.session_id,
                    }
            self._send(200, json.dumps(payload), "application/json; charset=utf-8")

        def do_POST(self):
            p = urlparse(self.path).path

            if p == "/rep":
                # triggers one rep if a squat session is active
                core.report_rep()
                return self._send(200, "REP_OK")

            if p == "/stop":
                core.try_stop_alarm()
                return self._send(200, "STOP_OK")

            return self._send(404, "Not Found")

        def log_message(self, format, *args):
            # silence default request logs (optional)
            return

    httpd = ThreadingHTTPServer((host, port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# -------------------------
# CLI
# -------------------------
HELP_TEXT = """
Nuka Squat Alarm — Help

You can control alarms using natural language (Chinese or English).

────────────────────────
1) Create an alarm
────────────────────────
Examples:
  - 帮我订一个明早七点的闹钟
  - 帮我订一个10分钟以后的闹钟
  - set an alarm tomorrow 7am
  - set an alarm at tomorrow 13:00
  - set an alarm in 2 minutes

Notes:
  - Relative time (e.g. "10 minutes later") is supported
  - Morning phrases like “明早 / tomorrow morning” are handled safely

────────────────────────
2) List alarms
────────────────────────
Examples:
  - list alarms
  - 列出闹钟
  - 我现在有哪些闹钟

Tip:
  - Each alarm has an ID (recommended for later operations)

────────────────────────
3) Cancel an alarm
────────────────────────
Best practice (recommended):
  - cancel alarm 3
  - 取消闹钟 3

Also supported (less precise):
  - cancel tomorrow morning alarm
  - 把明天早上的闹钟取消

────────────────────────
4) Update (change) an alarm
────────────────────────
Examples:
  - 把闹钟 3 改到明天 8 点
  - 把明早六点的闹钟改成八点
  - change alarm 3 to tomorrow 13:30
  - change tomorrow 6 to 8

────────────────────────
5) When alarm is ringing (squat mode)
────────────────────────
Commands:
  - rep     → record ONE squat (camera / MoveNet can call this automatically)
  - stop    → stop alarm (ONLY after all required squats are done)

Rules:
  - Alarm cannot be stopped until target squats are completed
  - Each squat triggers a short “ding” sound
  - Alarm fully stops only after all squats are finished

────────────────────────
6) Exit
────────────────────────
  - exit
  - quit
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

    # NEW: start local HTTP API for MoveNet integration
    httpd = start_http_api(core, host="127.0.0.1", port=8008)
    print("HTTP API: POST http://127.0.0.1:8008/rep  |  GET http://127.0.0.1:8008/status")

    print("Nuka MVP is running. Type /help for help.")
    last_lang = "en"

    try:
        while True:
            s = input(">>> ").strip()
            if not s:
                continue

            last_lang = detect_lang(s)
            core.default_lang = last_lang


            if s in ("/help", "help", "?"):
                print(HELP_TEXT)
                continue
            if s in ("exit", "quit"):
                break
            if s == "rep":
                # session language follows last command language (simple but effective for MVP)
                with core.lock:
                    if core.active:
                        core.active.lang = last_lang
                core.report_rep()
                continue
            if s == "stop":
                with core.lock:
                    if core.active:
                        core.active.lang = last_lang
                core.try_stop_alarm()
                continue

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

                # confirmation: NO LLM (fast + deterministic)
                human = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                if last_lang == "zh":
                    print(f"✅ 已创建闹钟 #{aid}：{human}（深蹲 {reps} 个解锁）。")
                else:
                    print(f"✅ Created alarm #{aid} at {human} (reps={reps}).")
                continue

            if it == "CANCEL_ALARM":
                which = intent.get("which", "")
                a = core.find_alarm_by_text(which)
                if not a:
                    print("❌ No matching alarm found.")
                    continue
                ok = core.cancel_alarm(a["id"])
                if last_lang == "zh":
                    print(f"✅ 已取消闹钟 #{a['id']}（{a['time_iso']}）。" if ok else "❌ 取消失败。")
                else:
                    print(f"✅ Canceled alarm #{a['id']} ({a['time_iso']})." if ok else "❌ Cancel failed.")
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
                human = new_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                if last_lang == "zh":
                    print(f"✅ 已更新闹钟 #{a['id']} → {human}。" if ok else "❌ 更新失败。")
                else:
                    print(f"✅ Updated alarm #{a['id']} -> {human}." if ok else "❌ Update failed.")
                continue

            if it == "STOP_ALARM":
                with core.lock:
                    if core.active:
                        core.active.lang = last_lang
                core.try_stop_alarm()
                continue

            # fallback chat: deterministic (no LLM here in MVP)
            if last_lang == "zh":
                print("我可以帮你设置/取消/修改闹钟。输入 /help 查看用法。")
            else:
                print("I can set/cancel/update alarms. Type /help.")

    except KeyboardInterrupt:
        pass
  
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        core.shutdown()
        print("Bye.")



if __name__ == "__main__":
    main()
