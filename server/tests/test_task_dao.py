
"""Unit tests for TaskDAO component."""
from unittest.mock import patch

import pytest

from server.server_defines import TaskStatus
from server.task_dao import TaskDAO
from server.tests.neo4j_mocks import MockResult


class MockSession:
    def __init__(self):
        self.calls = []

    def run(self, query, params=None):
        self.calls.append((query, params))
        q = " ".join((query or "").split()).lower()
        params = params or {}

        # For create/update operations we should return a non-None record
        if q.lower().startswith("create") and "processingstatus" in q and "return" in q:
            # Return a dict emulating a Record so dao.create_task_record sees non-None
            return MockResult(single_record={"request_id": params.get("request_id")})
        if "match (p:processingstatus {request_id:" in q and "return p" in q:
            rid = params.get("request_id")
            if rid == "ok":
                return MockResult(single_record={
                    "request_id": "ok",
                    "task_type": "X",
                    "status": "queued",
                    "created_at": 1,
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None,
                    "query": None,
                    "event_type": None,
                    "src_path": None,
                    "dest_path": None,
                    "is_directory": False,
                })
            return MockResult(single_record=None)
        if "delete p" in q and "return count(p) as deleted_count" in q:
            rid = params.get("request_id")
            deleted_count: int
            match rid:
                case "ok":
                    deleted_count = 1
                case None:
                    deleted_count = 5
                case _:
                    deleted_count = 0

            return MockResult(single_record={"deleted_count": deleted_count})
        if "match (p:processingstatus) set p.status" in q:
            return MockResult(records=[
                {"request_id": "a", "task_type": "T", "status": params.get("status", "queued"),
                 "created_at": 2, "started_at": None, "completed_at": None, "result": None, "error": None,
                 "query": None, "event_type": None, "src_path": None, "dest_path": None, "is_directory": False}
            ])
        if "order by p.created_at desc" in q and "limit $limit" in q:
            return MockResult(records=[
                {"request_id": "a", "task_type": "T", "status": "queued", "created_at": 2, "started_at": None,
                 "completed_at": None, "result": None, "error": None, "query": None, "event_type": None,
                 "src_path": None, "dest_path": None, "is_directory": False}
            ])
        if "where p.created_at < $timestamp" in q and "return count(p) as deleted_count" in q:
            return MockResult(single_record={"deleted_count": 3})
        if "return p.status as status" in q and "count(*) as count" in q:
            return MockResult(records=[
                {"status": "queued", "task_type": "A", "count": 2},
                {"status": "running", "task_type": "B", "count": 1},
            ])

        # Default: no record
        return MockResult(single_record=None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

class MockDriver:
    """Mock Neo4j driver."""

    def __init__(self):
        self.session_obj = MockSession()

    def session(self, **kwargs):
        return self.session_obj


@pytest.fixture
def mock_driver():
    """Fixture providing a mock Neo4j driver."""
    return MockDriver()


@pytest.fixture
def dao(mock_driver):
    """Fixture providing a TaskDAO instance with mocked dependencies."""
    with patch.object(TaskDAO, 'create_indexes', return_value=True):
        return TaskDAO(mock_driver)


# Valid test cases
def test_create_task_record_success(dao):
    """Test successful task record creation."""
    query = """
    CREATE (q:ProcessingStatus:QueryTask {
        request_id: $request_id,
        project_id: $project_id,
        task_type: $task_type,
        query: $query,
    })
    RETURN q.request_id as request_id
    """

    parameters = {
        "request_id": '1234',
        "project_id": '5678',
        "query": 'return 1',
        "task_type": 'query'
    }

    result = dao.create_task_record(
        query,
        parameters
    )
    assert result is True


def test_get_task_record_found(dao):
    """Test retrieving an existing task record."""
    record = dao.get_task_record("ok")
    assert record is not None
    assert record["request_id"] == "ok"


def test_get_tasks_by_status_returns_results(dao):
    """Test getting tasks by status."""
    tasks = dao.get_tasks_by_status(TaskStatus.QUEUED, limit=10)
    assert isinstance(tasks, list)
    assert len(tasks) > 0
    assert tasks[0]["status"] == "queued"


def test_get_recent_tasks_returns_results(dao):
    """Test getting recent tasks."""
    tasks = dao.get_recent_tasks(limit=5)
    assert isinstance(tasks, list)
    assert len(tasks) > 0


def test_delete_task_record_success(dao):
    """Test successful task record deletion."""
    result = dao.delete_task_record("ok")
    assert result is True


def test_cleanup_old_tasks_returns_count(dao):
    """Test cleaning up old tasks."""
    deleted_count = dao.cleanup_old_tasks(older_than_timestamp=1000000000.0)
    assert deleted_count == 5


def test_get_task_stats_returns_aggregates(dao):
    """Test getting task statistics."""
    stats = dao.get_task_stats()
    assert "total" in stats
    assert "by_status" in stats
    assert "by_type" in stats
    assert stats["total"] == 3
    assert stats["by_status"]["queued"] == 2
    assert stats["by_type"]["A"] == 2


def test_update_task_record_success(dao):
    """Test successful task record update."""
    result = dao.update_task_record(
        "MATCH (p:ProcessingStatus) SET p.status = $status RETURN p",
        {"request_id": "existing", "status": "running"}
    )
    assert result is True


# Invalid test cases
def test_get_task_record_not_found(dao):
    """Test retrieving a non-existent task record."""
    record = dao.get_task_record("nonexistent")
    assert record is None


def test_delete_task_record_not_found(dao):
    """Test deleting a non-existent task record."""
    result = dao.delete_task_record("nonexistent")
    assert result is False


def test_create_task_record_with_invalid_query(dao):
    """Test creating task record with query that returns None."""
    with patch.object(dao.driver.session_obj, 'run', return_value=MockResult(single_record=None)):
        result = dao.create_task_record("INVALID QUERY", {"request_id": "test"})
        assert result is False


def test_update_task_record_with_invalid_query(dao):
    """Test updating task record with query that returns None."""
    with patch.object(dao.driver.session_obj, 'run', return_value=MockResult(single_record=None)):
        result = dao.update_task_record("INVALID QUERY", {"request_id": "test"})
        assert result is False


def test_get_tasks_by_status_with_exception(dao):
    """Test getting tasks by status when database error occurs."""
    with patch.object(dao.driver.session_obj, 'run', side_effect=Exception("Database error")):
        tasks = dao.get_tasks_by_status(TaskStatus.PROCESSING)
        assert tasks == []


def test_get_task_stats_with_exception(dao):
    """Test getting task stats when database error occurs."""
    with patch.object(dao.driver.session_obj, 'run', side_effect=Exception("Database error")):
        stats = dao.get_task_stats()
        assert stats == {"total": 0, "by_status": {}, "by_type": {}}


def test_cleanup_old_tasks_with_exception(dao):
    """Test cleanup when database error occurs."""
    with patch.object(dao.driver.session_obj, 'run', side_effect=Exception("Database error")):
        count = dao.cleanup_old_tasks(1000000000.0)
        assert count == 0
