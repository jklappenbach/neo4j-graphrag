"""Unit tests for WebSocketManager component."""
import asyncio
import json

import pytest

from server.websocket_manager import WebSocketManagerImpl


class MockWebSocket:
    """Mock WebSocket for testing."""
    def __init__(self):
        self.accepted = False
        self.sent_messages = []
        self.closed = False
        self._should_fail = False
    
    async def accept(self):
        self.accepted = True
    
    async def send_text(self, text: str):
        if self._should_fail:
            raise Exception("Send failed")
        self.sent_messages.append(text)
    
    async def receive_text(self):
        await asyncio.sleep(0.01)
        return '{"type": "test"}'
    
    def set_fail_mode(self, should_fail: bool):
        self._should_fail = should_fail


@pytest.fixture
def ws_manager():
    """Fixture providing a WebSocketManager instance."""
    return WebSocketManagerImpl()


@pytest.fixture
def mock_ws():
    """Fixture providing a mock WebSocket."""
    return MockWebSocket()


# Valid test cases
@pytest.mark.asyncio
async def test_connect_websocket_success(ws_manager, mock_ws):
    """Test successfully connecting a WebSocket."""
    await ws_manager.connect(mock_ws, "conn-1")
    
    assert mock_ws.accepted is True
    assert "conn-1" in ws_manager._connections
    assert ws_manager._connections["conn-1"] is mock_ws


@pytest.mark.asyncio
async def test_disconnect_removes_connection(ws_manager, mock_ws):
    """Test disconnecting removes connection and request mappings."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    ws_manager.register_request("req-2", "conn-1")
    
    ws_manager.disconnect("conn-1")
    
    assert "conn-1" not in ws_manager._connections
    assert "req-1" not in ws_manager._request_to_connection
    assert "req-2" not in ws_manager._request_to_connection


@pytest.mark.asyncio
async def test_register_request_maps_correctly(ws_manager, mock_ws):
    """Test registering request creates correct mapping."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    
    assert ws_manager._request_to_connection["req-1"] == "conn-1"


@pytest.mark.asyncio
async def test_send_message_success(ws_manager, mock_ws):
    """Test successfully sending a message."""
    await ws_manager.connect(mock_ws, "conn-1")
    message = {"type": "test", "data": "hello"}
    
    result = await ws_manager.send_message("conn-1", message)
    
    assert result is True
    assert len(mock_ws.sent_messages) == 1
    sent = json.loads(mock_ws.sent_messages[0])
    assert sent["type"] == "test"
    assert sent["data"] == "hello"


@pytest.mark.asyncio
async def test_send_response_routes_to_connection(ws_manager, mock_ws):
    """Test sending response routes through request mapping."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    message = {"type": "result", "data": "response"}
    
    result = await ws_manager.send_response("req-1", message)
    
    assert result is True
    assert len(mock_ws.sent_messages) == 1
    sent = json.loads(mock_ws.sent_messages[0])
    assert sent["type"] == "result"
    assert sent["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_send_response_cleans_up_terminal_states(ws_manager, mock_ws):
    """Test terminal task states clean up request mappings."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    
    # Send completed message
    await ws_manager.send_response("req-1", {"type": "task_completed"})
    assert "req-1" not in ws_manager._request_to_connection
    
    # Test failed
    ws_manager.register_request("req-2", "conn-1")
    await ws_manager.send_response("req-2", {"type": "task_failed"})
    assert "req-2" not in ws_manager._request_to_connection
    
    # Test cancelled
    ws_manager.register_request("req-3", "conn-1")
    await ws_manager.send_response("req-3", {"type": "task_cancelled"})
    assert "req-3" not in ws_manager._request_to_connection


@pytest.mark.asyncio
async def test_send_response_preserves_non_terminal_mappings(ws_manager, mock_ws):
    """Test non-terminal states preserve request mappings."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    
    await ws_manager.send_response("req-1", {"type": "task_started"})
    
    # Mapping should still exist
    assert "req-1" in ws_manager._request_to_connection


# Invalid test cases
@pytest.mark.asyncio
async def test_send_message_connection_not_found(ws_manager):
    """Test sending message to non-existent connection."""
    result = await ws_manager.send_message("nonexistent", {"type": "test"})
    assert result is False


@pytest.mark.asyncio
async def test_send_response_no_request_mapping(ws_manager, mock_ws):
    """Test sending response when request mapping doesn't exist."""
    await ws_manager.connect(mock_ws, "conn-1")
    
    result = await ws_manager.send_response("unmapped-req", {"type": "test"})
    
    assert result is False
    assert len(mock_ws.sent_messages) == 0


@pytest.mark.asyncio
async def test_send_message_fails_and_disconnects(ws_manager, mock_ws):
    """Test send failure triggers disconnect."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    mock_ws.set_fail_mode(True)
    
    result = await ws_manager.send_message("conn-1", {"type": "test"})
    
    assert result is False
    assert "conn-1" not in ws_manager._connections
    assert "req-1" not in ws_manager._request_to_connection


@pytest.mark.asyncio
async def test_disconnect_nonexistent_connection(ws_manager):
    """Test disconnecting a connection that doesn't exist."""
    # Should not raise an error
    ws_manager.disconnect("nonexistent")
    assert "nonexistent" not in ws_manager._connections


@pytest.mark.asyncio
async def test_multiple_requests_same_connection(ws_manager, mock_ws):
    """Test multiple requests mapped to same connection."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    ws_manager.register_request("req-2", "conn-1")
    ws_manager.register_request("req-3", "conn-1")
    
    # Disconnect should remove all mappings
    ws_manager.disconnect("conn-1")
    
    assert "req-1" not in ws_manager._request_to_connection
    assert "req-2" not in ws_manager._request_to_connection
    assert "req-3" not in ws_manager._request_to_connection


@pytest.mark.asyncio
async def test_send_response_with_send_failure(ws_manager, mock_ws):
    """Test send_response when underlying send_message fails."""
    await ws_manager.connect(mock_ws, "conn-1")
    ws_manager.register_request("req-1", "conn-1")
    mock_ws.set_fail_mode(True)
    
    result = await ws_manager.send_response("req-1", {"type": "test"})
    
    assert result is False
    # Connection should be disconnected
    assert "conn-1" not in ws_manager._connections
