# Python
import json
import uuid
import pytest
import anyio
import trio

from types import SimpleNamespace

from server.websocket_manager import WebSocketManagerImpl


class FakeWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent = []
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        if self.closed:
            raise RuntimeError("Socket closed")
        # store parsed if JSON, else raw
        try:
            self.sent.append(json.loads(data))
        except Exception:
            self.sent.append(data)

    async def receive_text(self):
        return "Hey there!"

    async def close(self, code: int = 1000):
        self.closed = True


@pytest.mark.anyio
async def test_connect_sends_ws_connected_and_tracks_connection():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = str(uuid.uuid4())

    await mgr.connect(ws, cid)

    # accepted and stored
    assert ws.accepted is True
    # sent initial ws_connected message
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "ws_connected"
    assert ws.sent[0]["connection_id"] == cid
    # internal storage has the connection
    # (implementation has _connections dict)
    assert cid in mgr._connections


@pytest.mark.anyio
async def test_stop_is_idempotent_and_handles_no_connections():
    mgr = WebSocketManagerImpl()

    # Calling stop with nothing connected should not raise and should remain clean
    mgr.stop()
    await anyio.sleep(0)

    assert mgr._connections == {}  # noqa: SLF001
    assert mgr._receive_tasks == {}  # noqa: SLF001

    # Connect one, then stop twice
    ws = FakeWebSocket()
    cid = "idempotent-1"
    await mgr.connect(ws, cid)
    assert cid in mgr._connections  # noqa: SLF001

    mgr.stop()
    await anyio.sleep(0)
    # Second stop should be a no-op without exceptions
    mgr.stop()
    await anyio.sleep(0)

    assert cid not in mgr._connections  # noqa: SLF001
    assert cid not in mgr._receive_tasks  # noqa: SLF001

@pytest.mark.anyio
async def test_disconnect_removes_connection_and_mappings():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = "c-1"
    rid = "r-1"

    await mgr.connect(ws, cid)
    mgr.register_request(rid, cid)

    # sanity
    assert cid in mgr._connections  # noqa: SLF001
    assert mgr._request_to_connection.get(rid) == cid  # noqa: SLF001

    mgr.disconnect(cid)

    assert cid not in mgr._connections  # noqa: SLF001
    assert rid not in mgr._request_to_connection  # noqa: SLF001


@pytest.mark.anyio
async def test_register_request_and_send_response_routes_by_request_id():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = "c-2"
    rid = "r-2"

    await mgr.connect(ws, cid)
    mgr.register_request(rid, cid)

    payload = {"type": "task_started", "foo": "bar"}
    ok = await mgr.send_response(rid, payload)

    assert ok is True
    assert len(ws.sent) >= 2  # includes ws_connected + routed message
    routed = ws.sent[-1]
    assert routed["type"] == "task_started"
    assert routed["foo"] == "bar"
    assert routed["request_id"] == rid


@pytest.mark.anyio
async def test_send_response_for_terminal_state_cleans_mapping():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = "c-3"
    rid = "r-3"

    await mgr.connect(ws, cid)
    mgr.register_request(rid, cid)
    assert mgr._request_to_connection.get(rid) == cid  # noqa: SLF001

    # terminal state
    await mgr.send_response(rid, {"type": "task_completed"})

    assert rid not in mgr._request_to_connection  # noqa: SLF001


@pytest.mark.anyio
async def test_send_message_returns_false_if_connection_missing():
    mgr = WebSocketManagerImpl()
    ok = await mgr.send_message("missing-conn", {"type": "x"})
    assert ok is False


@pytest.mark.anyio
async def test_send_response_returns_false_if_no_mapping():
    mgr = WebSocketManagerImpl()
    ok = await mgr.send_response("unknown-request", {"type": "task_started"})
    assert ok is False


@pytest.mark.anyio
async def test_send_message_handles_send_failure_and_disconnects():
    mgr = WebSocketManagerImpl()

    class FailingWS(FakeWebSocket):
        async def send_text(self, data: str):
            raise RuntimeError("boom")

    ws = FailingWS()
    cid = "c-4"

    await mgr.connect(ws, cid)
    ok = await mgr.send_message(cid, {"type": "anything"})
    assert ok is False
    # connection should be removed after failure
    assert cid not in mgr._connections  # noqa: SLF001


@pytest.mark.anyio
async def test_handle_message_register_request_successful_acks():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = "c-5"

    await mgr.connect(ws, cid)
    await mgr.handle_message(cid, {"type": "register_request", "request_id": "r-5"})

    # Should send an ack message
    # first message was ws_connected; second is request_registered
    assert ws.sent[-1]["type"] == "request_registered"
    assert ws.sent[-1]["request_id"] == "r-5"
    assert mgr._request_to_connection.get("r-5") == cid  # noqa: SLF001


@pytest.mark.anyio
async def test_handle_message_register_request_missing_id_sends_error():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = "c-6"

    await mgr.connect(ws, cid)
    await mgr.handle_message(cid, {"type": "register_request"})

    # Error response appended after ws_connected
    err = ws.sent[-1]
    assert err["type"] == "error"
    assert err["success"] is False
    assert "Missing request_id" in err["error"]


@pytest.mark.anyio
async def test_handle_message_unknown_type_sends_error():
    mgr = WebSocketManagerImpl()
    ws = FakeWebSocket()
    cid = "c-7"

    await mgr.connect(ws, cid)
    await mgr.handle_message(cid, {"type": "unknown_thing"})

    err = ws.sent[-1]
    assert err["type"] == "error"
    assert err["success"] is False
    assert "Unsupported message type" in err["error"]
