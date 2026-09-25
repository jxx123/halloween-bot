#!/usr/bin/env python
"""Abstract agent layer: instructions in, event stream out.

Every backend implements `run(text, emit)`. `emit(event)` is called with dicts:
  {"type": "say",     "text": ...}   agent speaking to the user (drives face TTS)
  {"type": "thought", "text": ...}   agent inner monologue / progress
  {"type": "action",  "text": ...}   tool call / robot command summary
  {"type": "status",  "text": ...}   lifecycle (started/done/error)
Backends are selected by name via get_agent(); adding one = subclass + registry entry.
"""
import json
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

ROBOT_DIR = Path(__file__).resolve().parents[2]  # repo root (CLAUDE.md lives here)

_current_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None


def _register_proc(proc):
    global _current_proc
    with _current_lock:
        _current_proc = proc


def abort_current() -> bool:
    """Kill the in-flight agent subprocess, if any. Returns True if one was killed."""
    with _current_lock:
        proc = _current_proc
    if proc is None or proc.poll() is not None:
        return False
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    return True


class AgentBackend(ABC):
    name = "abstract"

    @abstractmethod
    def run(self, text: str, emit) -> None:
        """Handle one instruction (blocking; called on a worker thread)."""


class EchoAgent(AgentBackend):
    """No-LLM test agent: exercises the UI/face without spending tokens."""

    name = "echo"

    def run(self, text, emit):
        emit({"type": "thought", "text": f"pretending to think about: {text!r}"})
        time.sleep(1.5)
        emit({"type": "action", "text": "wiggling imaginary claws"})
        time.sleep(1.0)
        emit({"type": "say", "text": f"You said: {text}. Beep boop, happy to help!"})


class CliStreamAgent(AgentBackend):
    """Base for CLI coding agents that stream JSON lines (Claude Code, Codex)."""

    binary: str = ""

    def build_cmd(self, text: str) -> list[str]:
        raise NotImplementedError

    def handle_line(self, obj: dict, emit) -> None:
        raise NotImplementedError

    def run(self, text, emit):
        if not shutil.which(self.binary):
            emit({"type": "status", "text": f"error: '{self.binary}' CLI not found on PATH"})
            return
        cmd = self.build_cmd(text)
        emit({"type": "status", "text": f"started {self.name}"})
        proc = subprocess.Popen(
            cmd, cwd=ROBOT_DIR, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        _register_proc(proc)
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    emit({"type": "thought", "text": line[:300]})
                    continue
                try:
                    self.handle_line(obj, emit)
                except Exception as e:  # never let a parse hiccup kill the run
                    emit({"type": "thought", "text": f"(unparsed {type(e).__name__}) {str(obj)[:200]}"})
            code = proc.wait()
            if code < 0:  # killed by abort_current()
                emit({"type": "status", "text": "stopped by user"})
            else:
                emit({"type": "status", "text": "done" if code == 0 else f"error: exit {code}"})
        finally:
            if proc.poll() is None:
                proc.terminate()
            _register_proc(None)


class ClaudeCodeAgent(CliStreamAgent):
    """Headless Claude Code. cwd is claude_robot/, whose CLAUDE.md teaches robot control
    (ctl.py start/state/move/cam/stop), so instructions like "wave the right arm" work.
    Pass `persona` (a system-prompt suffix) to give the instance a voice-facing character;
    each instance tracks its own session_id so personas don't cross-continue each other."""

    binary = "claude"

    def __init__(self, name: str = "claude", persona: str | None = None, model: str = "opus"):
        self.name = name
        self.persona = persona
        self.model = model
        self.session_id = None

    def build_cmd(self, text):
        cmd = ["claude", "-p", text, "--model", self.model,
               "--output-format", "stream-json", "--verbose",
               "--allowedTools", "Bash", "Read"]
        if self.persona:
            cmd += ["--append-system-prompt", self.persona]
        if self.session_id:
            cmd += ["--resume", self.session_id]  # continue THIS persona's conversation
        return cmd

    def handle_line(self, obj, emit):
        if obj.get("session_id"):
            self.session_id = obj["session_id"]
        t = obj.get("type")
        if t == "assistant":
            for block in obj.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    emit({"type": "say", "text": block["text"].strip()})
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    detail = inp.get("command") or inp.get("file_path") or ""
                    emit({"type": "action", "text": f"{name}: {str(detail)[:160]}"})
                elif block.get("type") == "thinking" and block.get("thinking", "").strip():
                    emit({"type": "thought", "text": block["thinking"].strip()[:300]})
        elif t == "result":
            if obj.get("is_error"):
                emit({"type": "status", "text": f"error: {str(obj.get('result'))[:200]}"})


class CodexAgent(CliStreamAgent):
    """OpenAI Codex CLI backend (experimental — event shape parsed best-effort)."""

    name = "codex"
    binary = "codex"

    def build_cmd(self, text):
        return ["codex", "exec", "--json", text]

    def handle_line(self, obj, emit):
        msg = obj.get("msg", obj)
        t = msg.get("type", "")
        if "message" in t and msg.get("message"):
            emit({"type": "say", "text": str(msg["message"])[:500]})
        elif "exec" in t or "command" in t:
            emit({"type": "action", "text": str(msg.get("command", msg))[:160]})
        elif "reasoning" in t and msg.get("text"):
            emit({"type": "thought", "text": str(msg["text"])[:300]})


PERSONA_FILE = Path(__file__).resolve().parent / "persona.md"
_persona = PERSONA_FILE.read_text() if PERSONA_FILE.exists() else None

_REGISTRY = {a.name: a for a in (
    ClaudeCodeAgent(),                                    # technical console agent
    ClaudeCodeAgent(name="pumpkin", persona=_persona),    # kid-facing Halloween voice
    CodexAgent(),
    EchoAgent(),
)}


def get_agent(name: str) -> AgentBackend:
    return _REGISTRY.get(name, _REGISTRY["echo"])


def agent_names() -> list[str]:
    return list(_REGISTRY)
