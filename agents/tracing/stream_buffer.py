"""Thread-safe in-memory stream buffers for SSE thinking/reply channels.

这个模块做的事：给 LLM 的流式输出提供两个"写字板"。
  - thinking 通道：Planner 的思考过程（<thinking>...</thinking>）
  - reply 通道：Synthesizer 的最终回复

写（LLM 流式回调）：start → append → append → ... → finish
读（service.py 轮询）：  get → collect 增量 → 推 SSE

两个通道独立，用 Cursor 跟踪各自已消费到的位置，只取增量。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

# ===== 全局状态：所有 session 的流缓冲区都存在这个 dict 里 =====
_lock = Lock()
_sessions: dict[str, SessionStreams] = {}


# ===== 数据结构 =====

@dataclass
class StreamChannel:
    """一个流通道（thinking 或 reply），就像一个可写的字符串缓冲区。"""
    turn_id: str
    phase: str         # 前端展示用（"planner" / "synthesizer"）
    text: str = ""     # 累积的全部文本
    done: bool = False # LLM 已经写完了


@dataclass
class SessionStreams:
    """一个 session 的两个通道。"""
    thinking: StreamChannel | None = None
    reply: StreamChannel | None = None


# ===== 写操作：LLM 流式回调调用 =====

def reset_session_streams(session_id: str) -> None:
    """新一轮对话开始，清空旧缓冲区。"""
    with _lock:
        _sessions[session_id] = SessionStreams()


def start_thinking(session_id: str, turn_id: str, phase: str) -> None:
    """Planner 开始流式思考，初始化 thinking 通道。"""
    with _lock:
        streams = _sessions.setdefault(session_id, SessionStreams())
        streams.thinking = StreamChannel(turn_id=turn_id, phase=phase, text="", done=False)


def append_thinking(session_id: str, delta: str) -> None:
    """Planner 流式输出中，追加一段文字。"""
    if not delta:
        return
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).thinking
        if channel is None:
            return
        channel.text += delta


def finish_thinking(session_id: str) -> None:
    """Planner 思考完毕，标记 done。"""
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).thinking
        if channel is None:
            return
        channel.done = True


def start_reply(session_id: str, turn_id: str) -> None:
    """Synthesizer 开始流式回复，初始化 reply 通道。"""
    with _lock:
        streams = _sessions.setdefault(session_id, SessionStreams())
        streams.reply = StreamChannel(turn_id=turn_id, phase="synthesizer", text="", done=False)


def append_reply(session_id: str, delta: str) -> None:
    """Synthesizer 流式输出中，追加一段文字。"""
    if not delta:
        return
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).reply
        if channel is None:
            return
        channel.text += delta


def finish_reply(session_id: str) -> None:
    """Synthesizer 回复完毕，标记 done。"""
    with _lock:
        channel = _sessions.get(session_id, SessionStreams()).reply
        if channel is None:
            return
        channel.done = True


def get_session_streams(session_id: str) -> SessionStreams:
    """读取当前 session 的流缓冲区对象。"""
    with _lock:
        return _sessions.get(session_id, SessionStreams())


# ===== 读操作：service.py 轮询时调用 =====

@dataclass
class StreamEmitCursor:
    """消费游标 — 记录已读取到 thinking/reply 的第几个字。

    比如 thinking.text = "你好世界"，已推了前 2 个字：
      thinking_len = 2 → 下次从 text[2:] 开始取（"世界"）
    """
    thinking_len: int = 0
    reply_len: int = 0
    thinking_done: bool = False  # 已推送过 thinking done 事件
    reply_done: bool = False     # 已推送过 reply done 事件


def collect_stream_events(
    session_id: str,
    cursor: StreamEmitCursor,
) -> tuple[list[dict], StreamEmitCursor]:
    """从缓冲区取增量事件（service.py 轮询时调）。

    返回：
      events  — 需要推送给前端的增量事件列表
      cursor  — 更新后的游标（下次传入）

    比如：
      第 1 次调：thinking.text = "你好"，delta = "你好"，cursor.thinking_len = 2
      第 2 次调：thinking.text = "你好世界"，delta = "世界"，cursor.thinking_len = 4
    """
    events: list[dict] = []
    streams = get_session_streams(session_id)

    # --- thinking 通道 ---
    thinking = streams.thinking
    if thinking and len(thinking.text) > cursor.thinking_len:
        # 有新内容：取增量
        delta = thinking.text[cursor.thinking_len :]
        cursor.thinking_len = len(thinking.text)
        events.append({
            "type": "thinking",
            "turn_id": thinking.turn_id,
            "phase": thinking.phase,
            "delta": delta,        # 只推增量，不推全量
            "text": thinking.text, # 完整文本（前端可用做兜底）
            "done": thinking.done,
        })
    elif thinking and thinking.done and not cursor.thinking_done:
        # 已经结束了但之前没推过 done 事件：补推一次
        cursor.thinking_done = True
        events.append({
            "type": "thinking",
            "turn_id": thinking.turn_id,
            "phase": thinking.phase,
            "delta": "",
            "text": thinking.text,
            "done": True,
        })

    # --- reply 通道（逻辑同上） ---
    reply = streams.reply
    if reply and len(reply.text) > cursor.reply_len:
        delta = reply.text[cursor.reply_len :]
        cursor.reply_len = len(reply.text)
        events.append({
            "type": "reply_delta",
            "turn_id": reply.turn_id,
            "delta": delta,
            "reply": reply.text,
            "done": reply.done,
        })
    elif reply and reply.done and not cursor.reply_done:
        cursor.reply_done = True
        events.append({
            "type": "reply_delta",
            "turn_id": reply.turn_id,
            "delta": "",
            "reply": reply.text,
            "done": True,
        })

    return events, cursor
