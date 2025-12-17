"""Simple sound controller for Jetson Nano USB speaker.

The controller relies on ALSA utilities (`aplay` + `amixer`).  It can:
  * loop an alarm wav until stopped
  * fade alarm volume up/down
  * play a short ding for each squat rep

Usage:
    from NukaMotion.audio.sound_player import SoundController
    sound = SoundController()
    sound.start_alarm()
    sound.fade_alarm_to(0.3)
    sound.play_ding()
    sound.stop_alarm()

Make sure the WAV files exist:
    NukaMotion/audio/alarm.wav
    NukaMotion/audio/ding.wav

`device` should match `aplay -l` output, e.g. "plughw:2,0" for card 2.
`mixer_card` and `mixer_control` should be set to valid `amixer` targets
if you want smooth fades.  If amixer isn't available, fading silently
no-ops but alarm looping still works.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

AUDIO_DIR = Path(__file__).resolve().parent
DEFAULT_ALARM = AUDIO_DIR / "alarm.wav"
DEFAULT_DING = AUDIO_DIR / "ding.wav"


@dataclass
class SoundConfig:
    alarm_path: Path = DEFAULT_ALARM
    ding_path: Path = DEFAULT_DING
    device: str = "plughw:2,0"
    mixer_card: Optional[int] = 2
    mixer_control: str = "Speaker"
    fade_steps: int = 8
    fade_interval: float = 0.15


class SoundController:
    def __init__(self, config: SoundConfig | None = None):
        self.cfg = config or SoundConfig()
        self._alarm_thread: Optional[threading.Thread] = None
        self._alarm_stop = threading.Event()
        self._fade_lock = threading.Lock()
        self._current_level = 1.0
        self._fade_thread: Optional[threading.Thread] = None

    # -------------------- internal helpers --------------------
    def _resolve_audio_path(self, audio_path: Path) -> Path:
        if audio_path.exists():
            return audio_path
        # allow common alt extensions (e.g., wav vs mp3)
        candidates = []
        if audio_path.suffix:
            base = audio_path.with_suffix("")
            candidates.append(base.with_suffix(".wav"))
            candidates.append(base.with_suffix(".WAV"))
            candidates.append(base.with_suffix(".mp3"))
            candidates.append(base.with_suffix(".MP3"))
        else:
            candidates.extend([audio_path.with_suffix(ext) for ext in (".wav", ".WAV", ".mp3", ".MP3")])
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(audio_path)

    def _run_playback(self, audio_path: Path):
        real_path = self._resolve_audio_path(audio_path)
        cmd = self._build_play_cmd(real_path)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            pass

    def _build_play_cmd(self, audio_path: Path) -> list[str]:
        ext = audio_path.suffix.lower()
        return [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(audio_path),
        ]

    def _set_hw_volume(self, level: float):
        card = self.cfg.mixer_card
        if card is None:
            self._current_level = level
            return
        lvl = max(0, min(1.0, level))
        pct = int(round(lvl * 100))
        cmd = [
            "amixer",
            "-c",
            str(card),
            "set",
            self.cfg.mixer_control,
            f"{pct}%",
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            self._current_level = lvl
        except subprocess.CalledProcessError:
            # fall back to remembering level only
            self._current_level = lvl

    def _fade_worker(self, target: float, duration: float):
        with self._fade_lock:
            start = self._current_level
            steps = max(1, int(duration / self.cfg.fade_interval))
            for i in range(1, steps + 1):
                val = start + (target - start) * (i / steps)
                self._set_hw_volume(val)
                time.sleep(self.cfg.fade_interval)
            self._set_hw_volume(target)

    def fade_alarm_to(self, target: float, duration: float = 1.2):
        target = max(0.0, min(1.0, target))
        if self._fade_thread and self._fade_thread.is_alive():
            # let previous fade finish before starting new
            self._fade_thread.join(timeout=0.1)
        self._fade_thread = threading.Thread(
            target=self._fade_worker,
            args=(target, duration),
            daemon=True,
        )
        self._fade_thread.start()

    # -------------------- public API --------------------
    def start_alarm(self):
        if self._alarm_thread and self._alarm_thread.is_alive():
            return
        self._alarm_stop.clear()
        self._set_hw_volume(1.0)

        def loop():
            while not self._alarm_stop.is_set():
                self._run_playback(self.cfg.alarm_path)

        self._alarm_thread = threading.Thread(target=loop, daemon=True)
        self._alarm_thread.start()

    def stop_alarm(self):
        self._alarm_stop.set()
        if self._alarm_thread and self._alarm_thread.is_alive():
            self._alarm_thread.join(timeout=0.5)
        self._alarm_thread = None
        self._set_hw_volume(0.0)

    def play_ding(self):
        threading.Thread(
            target=self._run_playback,
            args=(self.cfg.ding_path,),
            daemon=True,
        ).start()


if __name__ == "__main__":
    sc = SoundController()
    print("Testing alarm for 3 seconds…")
    sc.start_alarm()
    time.sleep(1.5)
    sc.fade_alarm_to(0.3)
    time.sleep(1.5)
    sc.fade_alarm_to(1.0)
    time.sleep(1.0)
    sc.stop_alarm()
    print("Playing ding")
    sc.play_ding()
    time.sleep(1.0)
