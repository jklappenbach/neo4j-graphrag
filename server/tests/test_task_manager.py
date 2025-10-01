
"""Unit tests for TaskManagerImpl component."""
import asyncio
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from server.task_manager import TaskManagerImpl


class MockTask:
    """Mock Task for testing."""
    def __init__(self, request_id: str = "test-req-1"):
        self.request_id = request_id
        self.task_type = "test_task"
        self.is_finished = False
        self._enqueued = False
        self._started = False
        self._completed = False
        self._failed = False
        self._cancelled = False
        self._result = None
        self._error = None
    
    def get_task_type(self):
        return self.task_type
    
    def get_create_cypher(self):
        return ("CREATE (t:Task) RETURN t", {"request_id": self.request_id})
    
    def get_update_cypher(self):
        return ("MATCH (t:Task) SET t.status = 'updated' RETURN t", {"request_id": self.request_id})
    
    def to_dict(self):
        return {"request_id": self.request_id, "type": self.task_type}
    
    def enqueue(self):
        self._enqueued = True
    
    def start_processing(self):
        self._started = True
    
    def complete_with_result(self, result: Dict[str, Any]):
        self._completed = True
        self._result = result
        self.is_finished = True
    
    def fail_with_error(self, error: str):
        self._failed = True
        self._error = error
        self.is_finished = True
    
    def cancel(self):
        self._cancelled = True
        self.is_finished = True
    
    def execute(self):
        """Simulate task execution."""
        pass


class MockSession:
    """Mock Neo4j session for driver.session() support."""
    def __init__(self):
        self.run_calls = []
    
    def run(self, query: str, **params):
        self.run_calls.append((query, params))
        return Mock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class MockDriver:
    """Mock Neo4j driver with session support."""
    def __init__(self):
        self.session_obj = MockSession()
    
    def session(self, **kwargs):
        return self.session_obj


class MockDAO:
    """Mock TaskDAO for testing."""
    def __init__(self, driver=None):
        self.created_records = []
        self.updated_records = []
        self.driver = driver or MockDriver()
    
    def create_task_record(self, query: str, params: Dict[str, Any]) -> bool:
        self.created_records.append((query, params))
        return True
    
    def update_task_record(self, query: str, params: Dict[str, Any]) -> bool:
        self.updated_records.append((query, params))
        return True


@pytest.fixture
def mock_driver():
    """Fixture providing a mock Neo4j driver."""
    return MockDriver()


@pytest.fixture
def mock_dao(mock_driver):
    """Fixture providing a mock DAO."""
    return MockDAO(mock_driver)


@pytest.fixture
def notification_log():
    """Fixture to track notifications."""
    return []


@pytest.fixture
async def notifier(notification_log = None):
    """Fixture providing a mock websocket notifier."""
    async def _notifier(request_id: str, payload: Dict[str, Any]) -> None:
        notification_log.append((request_id, payload))
    return _notifier


@pytest.fixture
def task_manager(mock_dao, mock_driver, notifier):
    """Fixture providing TaskManagerImpl with mocked dependencies."""
    tm = TaskManagerImpl(mock_driver, notifier)
    tm._dao = mock_dao
    return tm


# Valid test cases
@pytest.mark.asyncio
def test_add_task_success(task_manager, mock_dao):
    """Test successfully adding a task."""
    task = MockTask("req-add-1")
    task_manager.add_task(task)
    
    # Allow async notification to complete
    asyncio.sleep(0)
    
    assert task._enqueued is True
    assert task.request_id in task_manager._records

@pytest.mark.asyncio
def test_complete_task_success(task_manager, mock_dao, notification_log):
    """Test successfully completing a task."""
    task = MockTask("req-complete-1")
    result = {"status": "done", "data": "result"}
    
    task_manager.complete_task(task, result)

    asyncio.sleep(0)

    assert task._completed is True
    assert task.is_finished is True
    assert task._result == result
    assert len(mock_dao.updated_records) == 1

@pytest.mark.asyncio
def test_fail_task_success(task_manager, mock_dao, notification_log):
    """Test successfully failing a task."""
    task = MockTask("req-fail-1")
    error_msg = "Something went wrong"
    
    task_manager.fail_task(task, error_msg)
    
    asyncio.sleep(0)
    
    assert task._failed is True
    assert task.is_finished is True
    assert task._error == error_msg
    assert len(mock_dao.updated_records) == 1

def test_cancel_task_success(task_manager, mock_dao, notification_log):
    """Test successfully cancelling a task."""
    task = MockTask("req-cancel-1")
    task_manager._records[task.request_id] = task
    
    task_manager.cancel_task(task.request_id)
    
    asyncio.sleep(0)
    
    assert task._cancelled is True
    assert task.is_finished is True
    assert len(mock_dao.updated_records) == 1

def test_get_task_success(task_manager):
    """Test retrieving a task."""
    task = MockTask("req-get-1")
    task_manager._records[task.request_id] = task
    
    retrieved = task_manager.get_task(task.request_id)
    
    assert retrieved is task
    assert retrieved.request_id == "req-get-1"


def test_list_active_tasks(task_manager):
    """Test listing active tasks filters out finished ones."""
    task1 = MockTask("req-active-1")
    task1.is_finished = False
    
    task2 = MockTask("req-active-2")
    task2.is_finished = True
    
    task3 = MockTask("req-active-3")
    task3.is_finished = False
    
    task_manager._records = {
        task1.request_id: task1,
        task2.request_id: task2,
        task3.request_id: task3
    }
    
    active = task_manager.list_active_tasks()
    
    assert len(active) == 2
    assert task1 in active
    assert task3 in active
    assert task2 not in active


# Invalid test cases
def test_get_task_not_found(task_manager):
    """Test retrieving a non-existent task."""
    result = task_manager.get_task("nonexistent-id")
    assert result is None


def test_cancel_task_not_found(task_manager, mock_dao):
    """Test cancelling a non-existent task."""
    result = task_manager.cancel_task("nonexistent-id")
    # Should return False and not update DAO
    assert result is False
    assert len(mock_dao.updated_records) == 0


def test_add_task_with_dao_failure(task_manager, mock_dao):
    """Test adding task when DAO fails."""
    mock_dao.create_task_record = Mock(side_effect=Exception("DAO error"))
    task = MockTask("req-fail-add")
    
    with pytest.raises(Exception, match="DAO error"):
        task_manager.add_task(task)


def test_start_task_with_dao_failure(task_manager, mock_dao):
    """Test starting task when DAO fails."""
    mock_dao.update_task_record = Mock(side_effect=Exception("DAO error"))
    task = MockTask("req-fail-start")
    
    # Should not raise, but return False
    result = task_manager.start_task(task)
    assert result is False


def test_complete_task_with_dao_failure(task_manager, mock_dao):
    """Test completing task when DAO fails."""
    mock_dao.update_task_record = Mock(side_effect=Exception("DAO error"))
    task = MockTask("req-fail-complete")
    
    result = task_manager.complete_task(task, {"data": "test"})
    assert result is False


def test_fail_task_with_dao_failure(task_manager, mock_dao):
    """Test failing task when DAO fails."""
    mock_dao.update_task_record = Mock(side_effect=Exception("DAO error"))
    task = MockTask("req-fail-fail")
    
    result = task_manager.fail_task(task, "error")
    assert result is False


def test_list_active_tasks_empty(task_manager):
    """Test listing active tasks when all are finished."""
    task1 = MockTask("fin-1")
    task1.is_finished = True
    task2 = MockTask("fin-2")
    task2.is_finished = True
    
    task_manager._records = {
        task1.request_id: task1,
        task2.request_id: task2
    }
    
    active = task_manager.list_active_tasks()
    assert len(active) == 0