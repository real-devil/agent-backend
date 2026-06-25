import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agents.runtime import get_workflow_snapshot, resume_agent_graph, run_agent_graph
from agents.tracing.stream_buffer import StreamEmitCursor, collect_stream_events, get_session_streams
from agents.tracing.user_sink import trace_for_turn as _trace_for_turn


def _current_turn_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    turn_id = snapshot.get("current_turn_id")
    if not turn_id:
        return None

    reply = snapshot.get("final_reply")
    if not reply:
        for message in reversed(snapshot.get("messages") or []):
            if message.get("role") == "assistant":
                reply = message.get("content", "")
                break

    return {
        "turn_id": turn_id,
        "user_message": snapshot.get("current_turn_user_message", ""),
        "started_at": snapshot.get("current_turn_started_at"),
        "status": snapshot.get("workflow_status"),
        "route_reason": snapshot.get("route_reason"),
        "review_decision": snapshot.get("review_decision"),
        "review_reason": snapshot.get("review_reason"),
        "pending_approval_group": snapshot.get("pending_approval_group"),
        "reply": reply or "",
        "workflow_plan": snapshot.get("workflow_plan") or [],
        "workflow_trace": _trace_for_turn(snapshot.get("workflow_trace") or [], str(turn_id)),
        "artifacts": snapshot.get("artifacts") or {},
        "metrics_summary": snapshot.get("metrics_summary") or {},
        "thinking_log": list(snapshot.get("turn_thinking_log") or []),
    }


def _conversation_turns(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    turns = list(snapshot.get("turn_history") or [])
    current_turn = _current_turn_from_snapshot(snapshot)
    if current_turn is None:
        return turns

    current_turn_id = str(current_turn["turn_id"])
    if any(str(turn.get("turn_id")) == current_turn_id for turn in turns):
        return turns

    return turns + [current_turn]


def _snapshot_to_session_state(session_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    current_turn_id = snapshot.get("current_turn_id")
    return {
        "session_id": session_id,
        "current_turn_id": current_turn_id,
        "workflow_status": snapshot.get("workflow_status"),
        "route_reason": snapshot.get("route_reason"),
        "current_group_index": snapshot.get("current_group_index"),
        "pending_approval_group": snapshot.get("pending_approval_group"),
        "review_decision": snapshot.get("review_decision"),
        "review_reason": snapshot.get("review_reason"),
        "artifacts": snapshot.get("artifacts"),
        "metrics_summary": snapshot.get("metrics_summary"),
        "workflow_trace": _trace_for_turn(snapshot.get("workflow_trace") or [], current_turn_id),
        "workflow_plan": snapshot.get("workflow_plan"),
        "conversation_turns": _conversation_turns(snapshot),
    }


from agents.tracing.user_sink import format_sse_event as _format_sse_event
from agents.tracing.user_sink import format_stream_sse as _format_stream_sse
from agents.tracing.user_sink import format_trace_sse as _format_trace_sse


def _streams_need_fast_poll(session_id: str) -> bool:
    streams = get_session_streams(session_id)
    thinking = streams.thinking
    reply = streams.reply
    return bool(
        (thinking is not None and not thinking.done)
        or (reply is not None and not reply.done)
    )


async def _yield_stream_events(
    session_id: str,
    cursor: StreamEmitCursor,
) -> tuple[list[str], StreamEmitCursor]:
    payloads: list[str] = []
    events, cursor = collect_stream_events(session_id, cursor)
    for event in events:
        payloads.append(_format_stream_sse(session_id, dict(event)))
    return payloads, cursor


def _snapshot_signature(snapshot: dict[str, Any]) -> str:
    artifacts = snapshot.get("artifacts") or {}
    artifact_digest = {
        key: {
            "artifact_type": value.get("artifact_type"),
            "summary": value.get("summary"),
            "confidence": value.get("confidence"),
        }
        for key, value in artifacts.items()
        if isinstance(value, dict)
    }

    return json.dumps(
        {
            "current_turn_id": snapshot.get("current_turn_id"),
            "workflow_status": snapshot.get("workflow_status"),
            "route_reason": snapshot.get("route_reason"),
            "current_group_index": snapshot.get("current_group_index"),
            "pending_approval_group": snapshot.get("pending_approval_group"),
            "review_decision": snapshot.get("review_decision"),
            "review_reason": snapshot.get("review_reason"),
            "artifact_digest": artifact_digest,
            "metrics_summary": snapshot.get("metrics_summary"),
            "workflow_plan": snapshot.get("workflow_plan"),
            "turn_ids": [turn.get("turn_id") for turn in _conversation_turns(snapshot)],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


async def run_agent_session(
    user_input: str,
    session_id: str | None = None,
    document_id: str | None = None,
) -> dict[str, str | None]:
    active_session_id = session_id or str(uuid4())
    reply = await run_agent_graph(
        user_input=user_input,
        session_id=active_session_id,
        document_id=document_id,
    )
    return {"reply": reply, "session_id": active_session_id}


async def stream_agent_session(
    user_input: str,
    session_id: str | None = None,
    document_id: str | None = None,
) -> AsyncIterator[str]:
    """SSE 流式执行 — 后台跑图 + 轮询推事件。

    数据流（所有 yield 都是实时推给前端的 SSE 消息）：
      ① 先推 session_id
      ② 图跑的过程中：推 trace 事件 → 推 thinking/reply 流式增量 → 推 state 快照
      ③ 图跑完：推剩余的 trace → 推剩余的流式数据 → 推最终 state → 推 final
    """
    active_session_id = session_id or str(uuid4())

    # 如果前端传了已有的 session_id，算出已有多少条旧 trace，只推新增的
    pre_trace_count = 0
    if session_id:
        pre_snapshot = await get_workflow_snapshot(active_session_id)
        pre_trace_count = len(pre_snapshot.get("workflow_trace") or [])

    # ========== 阶段 1：启动后台任务 ==========
    # 把 run_agent_graph 丢到事件循环后台，立刻返回，不阻塞
    # 图在后台跑，写入 checkpointer；我们前面轮询 checkpointer 观看
    task = asyncio.create_task(
        run_agent_graph(
            user_input=user_input,
            session_id=active_session_id,
            document_id=document_id,
        )
    )

    # 第一步：告诉前端 session_id
    yield _format_sse_event("session", {"session_id": active_session_id})

    emitted_trace_count = pre_trace_count  # 从这里开始，之前的旧事件不推
    last_signature = ""                     # 上次 state 快照的指纹，防重复推送
    stream_cursor = StreamEmitCursor()      # 记录 thinking/reply 已推到第几个字

    try:
        # ========== 阶段 2：轮询 - 图没跑完就一直推 ==========
        while not task.done():
            # 去 checkpointer 看最新的执行状态
            snapshot = await get_workflow_snapshot(active_session_id)

            # --- 推 trace 事件：step_started / tool_called / review_completed ... ---
            trace = snapshot.get("workflow_trace") or []
            while emitted_trace_count < len(trace):
                yield _format_trace_sse(active_session_id, trace[emitted_trace_count])
                emitted_trace_count += 1

            # --- 推 thinking / reply 流式增量（LLM 吐字的实时内容） ---
            stream_payloads, stream_cursor = await _yield_stream_events(active_session_id, stream_cursor)
            for payload in stream_payloads:
                yield payload

            # --- 如果整体状态变了，推一次完整快照（前端卡片展示用） ---
            current_signature = _snapshot_signature(snapshot)
            if current_signature != last_signature:
                yield _format_sse_event(
                    "state",
                    _snapshot_to_session_state(active_session_id, snapshot),
                )
                last_signature = current_signature

            # 活跃时 60ms 推一次（有流式内容），空闲时 400ms（LLM 还在调）
            await asyncio.sleep(0.06 if (stream_payloads or _streams_need_fast_poll(active_session_id)) else 0.4)

        # ========== 阶段 3：图跑完，收尾 ==========
        reply = await task  # 拿 run_agent_graph 的返回值（final_reply）

        # 再查一次，把最后可能漏掉的 trace / streaming / state 全部推出去
        snapshot = await get_workflow_snapshot(active_session_id)

        trace = snapshot.get("workflow_trace") or []
        while emitted_trace_count < len(trace):
            yield _format_trace_sse(active_session_id, trace[emitted_trace_count])
            emitted_trace_count += 1

        stream_payloads, stream_cursor = await _yield_stream_events(active_session_id, stream_cursor)
        for payload in stream_payloads:
            yield payload

        yield _format_sse_event(
            "state",
            _snapshot_to_session_state(active_session_id, snapshot),
        )
        # 最后一条：event: final，前端收到后关闭 EventSource
        yield _format_sse_event(
            "final",
            {
                "session_id": active_session_id,
                "reply": reply,
                "turn_id": snapshot.get("current_turn_id"),
            },
        )
    except Exception as exc:
        # 出了异常 → 取消后台任务 → 推 event: error → 前端展示错误提示
        if not task.done():
            task.cancel()
        yield _format_sse_event(
            "error",
            {
                "session_id": active_session_id,
                "detail": str(exc),
            },
        )


async def resume_agent_session(
    session_id: str,
    approval_response: str,
    user_input: str | None = None,
) -> dict[str, str | None]:
    reply = await resume_agent_graph(
        session_id=session_id,
        approval_response=approval_response,
        user_input=user_input,
    )
    return {"reply": reply, "session_id": session_id}


async def stream_resume_agent_session(
    session_id: str,
    approval_response: str,
    user_input: str | None = None,
) -> AsyncIterator[str]:
    task = asyncio.create_task(
        resume_agent_graph(
            session_id=session_id,
            approval_response=approval_response,
            user_input=user_input,
        )
    )

    yield _format_sse_event("session", {"session_id": session_id})

    emitted_trace_count = 0
    snapshot = await get_workflow_snapshot(session_id)
    trace = snapshot.get("workflow_trace") or []
    emitted_trace_count = len(trace)
    last_signature = _snapshot_signature(snapshot)
    stream_cursor = StreamEmitCursor()

    try:
        while not task.done():
            snapshot = await get_workflow_snapshot(session_id)
            trace = snapshot.get("workflow_trace") or []
            while emitted_trace_count < len(trace):
                yield _format_trace_sse(session_id, trace[emitted_trace_count])
                emitted_trace_count += 1

            stream_payloads, stream_cursor = await _yield_stream_events(session_id, stream_cursor)
            for payload in stream_payloads:
                yield payload

            current_signature = _snapshot_signature(snapshot)
            if current_signature != last_signature:
                yield _format_sse_event(
                    "state",
                    _snapshot_to_session_state(session_id, snapshot),
                )
                last_signature = current_signature

            await asyncio.sleep(0.06 if (stream_payloads or _streams_need_fast_poll(session_id)) else 0.4)

        reply = await task
        snapshot = await get_workflow_snapshot(session_id)
        trace = snapshot.get("workflow_trace") or []
        while emitted_trace_count < len(trace):
            yield _format_trace_sse(session_id, trace[emitted_trace_count])
            emitted_trace_count += 1

        stream_payloads, stream_cursor = await _yield_stream_events(session_id, stream_cursor)
        for payload in stream_payloads:
            yield payload

        yield _format_sse_event(
            "state",
            _snapshot_to_session_state(session_id, snapshot),
        )
        yield _format_sse_event(
            "final",
            {
                "session_id": session_id,
                "reply": reply,
                "turn_id": snapshot.get("current_turn_id"),
            },
        )
    except Exception as exc:
        if not task.done():
            task.cancel()
        yield _format_sse_event(
            "error",
            {
                "session_id": session_id,
                "detail": str(exc),
            },
        )


async def get_agent_session_state(session_id: str) -> dict[str, Any]:
    snapshot = await get_workflow_snapshot(session_id)
    return _snapshot_to_session_state(session_id, snapshot)
