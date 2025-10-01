"""Unit tests for GraphRagManagerImpl component."""
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest

from server.graph_rag_manager import GraphRagManagerImpl
from server.server_defines import Project
from server.tests.neo4j_mocks import MockResult


class MockSession:
    def __init__(self, driver, database: str | None = None):
        self._driver = driver
        self.database = database or "neo4j"
        self.calls = []

    def run(self, query: str, **params):
        self.calls.append((self.database, query, params or {}))
        q = " ".join((query or "").split()).lower()

        # Handle database management in "system" database
        if self.database == "system":
            # CREATE DATABASE $db IF NOT EXISTS
            if "create database $db if not exists" in q:
                db_name = params.get("db")
                if db_name:
                    self._driver._databases.add(db_name)
                return MockResult(single_record={"ok": True})
            # DROP DATABASE $db IF EXISTS
            if "drop database $db if exists" in q:
                db_name = params.get("db")
                if db_name and db_name in self._driver._databases:
                    self._driver._databases.remove(db_name)
                return MockResult(single_record={"ok": True})
            return MockResult(single_record={"ok": True})

        # In non-system DBs, simulate "database does not exist"
        if self.database not in self._driver._databases:
            raise RuntimeError(f"Database does not exist: {self.database}")

        # Generic success for schema and project catalog ops
        return MockResult(single_record={"ok": True})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

class MockDriver:
    def __init__(self):
        self._databases = set()  # holds created database names
        self._sessions = {}

    def session(self, database: str | None = None):
        db = database or "neo4j"
        if db not in self._sessions:
            self._sessions[db] = MockSession(self, database=db)
        return self._sessions[db]

class MockTaskManager:
    """Mock TaskManager for testing."""
    def __init__(self):
        self.added_tasks = []
        self.cancelled_tasks = []
    
    def add_task(self, task):
        self.added_tasks.append(task)
    
    def cancel_task(self, request_id: str):
        self.cancelled_tasks.append(request_id)


class MockProjectManager:
    """Mock ProjectManager for testing."""
    def __init__(self):
        self._projects = {}
    
    def create_project(self, project: Project) -> Dict[str, Any]:
        self._projects[project.project_id] = project
        return {
            "project_id": project.project_id,
            "name": project.name,
            "source_roots": project.source_roots,
            "args": project.args
        }
    
    def get_project(self, project_id: str) -> Project:
        return self._projects.get(project_id)
    
    def list_projects(self) -> List[Project]:
        return list(self._projects.values())
    
    def update_project(self, project_id: str, **kwargs) -> Dict[str, Any]:
        if project_id not in self._projects:
            raise ValueError(f"Project {project_id} not found")
        project = self._projects[project_id]
        for key, val in kwargs.items():
            if val is not None and hasattr(project, key):
                setattr(project, key, val)
        return {
            "project_id": project.project_id,
            "name": project.name,
            "source_roots": project.source_roots
        }
    
    def delete_project(self, project_id: str) -> Dict[str, Any]:
        if project_id in self._projects:
            del self._projects[project_id]
            return {"project_id": project_id, "deleted": True}
        return {"deleted": False}


@pytest.fixture
def mock_driver():
    """Fixture providing a mock Neo4j driver."""
    return MockDriver()


@pytest.fixture
def mock_task_manager():
    """Fixture providing a mock TaskManager."""
    return MockTaskManager()


@pytest.fixture
def graph_rag_manager(mock_driver, mock_task_manager):
    """Fixture providing GraphRagManagerImpl with mocked dependencies."""
    with patch('server.graph_rag_manager.ProjectManagerImpl', return_value=MockProjectManager()):
        with patch('server.graph_rag_manager.GraphDatabase'):
            manager = GraphRagManagerImpl(mock_driver, mock_task_manager)
            manager._project_mgr = MockProjectManager()
            return manager


# Valid test cases
def test_create_project_success(graph_rag_manager, mock_task_manager):
    """Test successfully creating a project."""
    project = Project(name="TestProject", source_roots=["/test/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    assert project.project_id in graph_rag_manager._project_entries
    created = graph_rag_manager._project_mgr.get_project(project.project_id)
    assert created is not None
    assert created.name == "TestProject"


def test_get_project_success(graph_rag_manager):
    """Test successfully retrieving a project."""
    project = Project(name="GetProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    retrieved = graph_rag_manager.get_project(project.project_id)
    
    assert retrieved is not None
    assert retrieved.name == "GetProject"


def test_list_projects_success(graph_rag_manager):
    """Test successfully listing projects."""
    project1 = Project(name="Project1", source_roots=["/src1"], args={})
    project2 = Project(name="Project2", source_roots=["/src2"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project1)
            graph_rag_manager.create_project(project2)
    
    projects = graph_rag_manager.list_projects()
    
    assert len(projects) == 2
    names = [p.name for p in projects]
    assert "Project1" in names
    assert "Project2" in names


def test_update_project_success(graph_rag_manager):
    """Test successfully updating a project."""
    project = Project(name="UpdateProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    with patch.object(graph_rag_manager, 'sync_project'):
        updated = graph_rag_manager.update_project(
            project.project_id,
            name="UpdatedName",
            source_roots=["/newsrc"]
        )
    
    assert updated["name"] == "UpdatedName"
    assert updated["source_roots"] == ["/newsrc"]


def test_delete_project_success(graph_rag_manager):
    """Test successfully deleting a project."""
    project = Project(name="DeleteProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    result = graph_rag_manager.delete_project(project.project_id)
    
    assert result["deleted"] is True
    assert project.project_id not in graph_rag_manager._project_entries


def test_query_enqueues_task(graph_rag_manager, mock_task_manager):
    """Test query operation enqueues a task."""
    project = Project(name="QueryProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    result = graph_rag_manager.query("req-1", project.project_id, "What is this?", {})
    
    assert result["ok"] is True
    assert len(mock_task_manager.added_tasks) > 0


def test_cancel_query_success(graph_rag_manager, mock_task_manager):
    """Test successfully cancelling a query."""
    result = graph_rag_manager.cancel_query("req-cancel-1")
    
    assert result["ok"] is True
    assert "req-cancel-1" in mock_task_manager.cancelled_tasks


def test_list_documents_enqueues_task(graph_rag_manager, mock_task_manager):
    """Test list_documents enqueues a task."""
    project = Project(name="ListDocProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    result = graph_rag_manager.list_documents("req-list-1", project.project_id)
    
    assert result["ok"] is True
    assert len(mock_task_manager.added_tasks) > 0


def test_refresh_documents_enqueues_task(graph_rag_manager, mock_task_manager):
    """Test refresh_documents enqueues a task."""
    project = Project(name="RefreshProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    result = graph_rag_manager.refresh_documents("req-refresh-1", project.project_id)
    
    assert result["ok"] is True
    assert len(mock_task_manager.added_tasks) > 0


# Invalid test cases
def test_get_project_not_found(graph_rag_manager):
    """Test retrieving a non-existent project."""
    result = graph_rag_manager.get_project("nonexistent-id")
    assert result is None


def test_delete_project_not_found(graph_rag_manager):
    """Test deleting a non-existent project."""
    result = graph_rag_manager.delete_project("nonexistent-id")
    assert result["deleted"] is False


def test_update_project_not_found(graph_rag_manager):
    """Test updating a non-existent project."""
    with pytest.raises(ValueError, match="not found"):
        graph_rag_manager.update_project("nonexistent-id", name="NewName")


def test_create_project_duplicate(graph_rag_manager):
    """Test creating a project that already exists."""
    project = Project(name="DupProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    # Set up duplicate entry
    graph_rag_manager._project_entries[project.project_id] = Mock()
    
    # Should raise exception
    with pytest.raises(Exception, match="already exists"):
        graph_rag_manager.create_project(project)


def test_query_invalid_project(graph_rag_manager):
    """Test query with invalid project ID."""
    result = graph_rag_manager.query("req-1", "invalid-proj-id", "query", {})
    # Should still return ok since task is enqueued, but will fail during execution
    assert result["ok"] is True


def test_handle_query_invalid_project_raises(graph_rag_manager):
    """Test handle_query with invalid project raises exception."""
    with pytest.raises(Exception, match="Invalid project ID"):
        graph_rag_manager.handle_query("req-1", "invalid-id", "query", {})


def test_sync_project_invalid_project_raises(graph_rag_manager):
    """Test sync_project with invalid project raises exception."""
    with pytest.raises(ValueError, match="not found"):
        graph_rag_manager.sync_project("invalid-id", force=False)


def test_handle_list_documents_invalid_project_raises(graph_rag_manager):
    """Test handle_list_documents with invalid project raises exception."""
    with pytest.raises(ValueError, match="not found"):
        graph_rag_manager.handle_list_documents("invalid-id")


def test_handle_add_path_nonexistent_path(graph_rag_manager):
    """Test handle_add_path with non-existent path."""
    project = Project(name="PathProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        graph_rag_manager.create_project(project)
    
    # Should handle gracefully and not raise
    nonexistent_path = Path("/nonexistent/path/file.py")
    with patch('server.graph_rag_manager._ProjectEntry'):
        graph_rag_manager.handle_add_path(project.project_id, nonexistent_path)


def test_handle_delete_path_nonexistent_raises(graph_rag_manager):
    """Test handle_delete_path with non-existent path raises FileNotFoundError."""
    project = Project(name="DelPathProject", source_roots=["/src"], args={})
    
    with patch.object(graph_rag_manager, '_create_project_observers'):
        with patch.object(graph_rag_manager, 'handle_add_path'):
            graph_rag_manager.create_project(project)
    
    nonexistent_path = Path("/nonexistent/path/file.py")
    with patch('server.graph_rag_manager._ProjectEntry'):
        with pytest.raises(FileNotFoundError):
            graph_rag_manager.handle_delete_path(project.project_id, nonexistent_path)
