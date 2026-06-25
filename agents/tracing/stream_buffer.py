"""Thread-safe in-memory stream buffers for SSE thinking/reply channels."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

_lock = Lock()
_sessions: dict[str, SessionStreams] = {}


@dataclass
class StreamChannel:
    turn_id: str
    phase: str
    text: str = ""
    done: bool = False


@dataclass
class SessionStreams:
    thinking: StreamChannel | None = None
    reply: StreamChannel | None = None


def reset_session_streams(session_id: str) -> None:
    with _lock:
        _sessions[session_id] = SessionStreams()


def start_thinking(session_id: str, turn_id: str, phase: str) -> None:
    with _lock:
        streams = _sessions.setdefault(session_id, SessionStreams())
        streams.thinking = StreamChannel(turn_id=turn_id, phase=phase, text="", done=False)


def append_thinking(session_id: str, delta: str) -> None:
    if not delta:
        return
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).thinking
        if channel is None:
            return
        channel.text += delta


def finish_thinking(session_id: str) -> None:
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).thinking
        if channel is None:
            return
        channel.done = True


def start_reply(session_id: str, turn_id: str) -> None:
    with _lock:
        streams = _sessions.setdefault(session_id, SessionStreams())
        streams.reply = StreamChannel(turn_id=turn_id, phase="synthesizer", text="", done=False)


def append_reply(session_id: str, delta: str) -> None:
    if not delta:
        return
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).reply
        if channel is None:
            return
        channel.text += delta


def finish_reply(session_id: str) -> None:
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).reply
        if channel is None:
            return
        channel.done = True


def get_session_streams(session_id: str) -> SessionStreams:
    with _lock:
        return _sessions.get(session_id, SessionStreams())


@dataclass
class StreamEmitCursor:
    thinking_len: int = 0
    reply_len: int = 0
    thinking_done: bool = False
    reply_done: bool = False


def collect_stream_events(
    session_id: str,
    cursor: StreamEmitCursor,
) -> tuple[list[dict], StreamEmitCursor]:
    events: list[dict] = []
    streams = get_session_streams(session_id)

    thinking = streams.thinking
    if thinking and len(thinking.text) > cursor.thinking_len:
        delta = thinking.text[cursor.thinking_len :]
        cursor.thinking_len = len(thinking.text)
        events.append(
            {
                "type": "thinking",
                "turn_id": thinking.turn_id,
                "phase": thinking.phase,
                "delta": delta,
                "text": thinking.text,
                "done": thinking.done,
            }
        )
    elif thinking and thinking.done and not cursor.thinking_done:
        cursor.thinking_done = True
        events.append(
            {
                "type": "thinking",
                "turn_id": thinking.turn_id,
                "phase": thinking.phase,
                "delta": "",
                "text": thinking.text,
                "done": True,
            }
        )

    reply = streams.reply
    if reply and len(reply.text) > cursor.reply_len:
        delta = reply.text[cursor.reply_len :]
        cursor.reply_len = len(reply.text)
        events.append(
            {
                "type": "reply_delta",
                "turn_id": reply.turn_id,
                "delta": delta,
                "reply": reply.text,
                "done": reply.done,
            }
        )
    elif reply and reply.done and not cursor.reply_done:
        cursor.reply_done = True
        events.append(
            {
                "type": "reply_delta",
                "turn_id": reply.turn_id,
                "delta": "",
                "reply": reply.text,
                "done": True,
            }
        )

    return events, cursor
