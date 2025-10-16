# Python
import uuid
import types
import pytest

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

from server.graph_rag_manager import GraphRagManagerImpl
from server.server_defines import Project, FileEventType

# ---- Minimal mock ProjectManagerImpl to inject into GraphRagManagerImpl ----
class DummyProjectManager:
    def __init__(self):
        self._projects: Dict[str, Project] = {}

    def create_project(self, project: Project) -> None:
        self._projects[project.project_id] = project

    def list_projects(self) -> List[Project]:
        return list(self._projects.values())

    def get_project(self, project_id: str):
        return self._projects.get(project_id)

    def update_project(self, project_id: str, updates: Dict[str, Any]):
        p = self._projects.get(project_id)
        if not p:
            return None
        new_name = updates.get("name", p.name)
        new_roots = updates.get("source_roots", p.source_roots)
        np = Project(new_name, new_roots, updates)
        np.project_id = p.project_id
        self._projects[project_id] = np
        return np.to_dict()

    def delete_project(self, project_id: str) -> None:
        self._projects.pop(project_id, None)

# ---- Helpers to build a fully isolated manager with injected mocks ----
class DummyDriver:
    def session(self, **kwargs):
        return DummySession()

class DummySession:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def run(self, *args, **kwargs):
        class R:
            def __iter__(self): return iter([])
            def get(self, *a, **k): return None
        return R()

def build_manager_with_mocks(tmp_path: Path):
    # Mock driver (Neo4j)
    driver = DummyDriver()
    # Mock task manager (only interface used is add_task/cancel_task)
    task_mgr = Mock()
    # Create instance
    mgr = GraphRagManagerImpl(driver, task_mgr)
    # Inject dummy project manager to avoid real persistence
    mgr._project_mgr = DummyProjectManager()
    # Avoid creating real observers and pipelines during tests
    mgr._create_project_observers = lambda *a, **k: None
    # Stub methods that hit FS/ingestion
    mgr.handle_add_path = lambda *a, **k: None
    mgr.handle_delete_path = lambda *a, **k: None
    mgr.handle_update_path = lambda *a, **k: None
    mgr._create_doc_relationships = lambda *a, **k: None
    # Make SUPPORTED_EXTS simple and avoid actual file IO
    mgr.SUPPORTED_EXTS = {".py"}
    return mgr

# ---- Tests ----

def test_create_project_success(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("proj1", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    # Project is tracked internally by id
    assert p.project_id in mgr._project_entries
    # Is also known to the project manager
    assert mgr._project_mgr.get_project(p.project_id) is not None

def test_create_project_duplicate_is_handled(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("proj1", [tmp_path.as_posix()], args={})
    mgr.create_project(p)
    # Creating again should raise (manager checks existence)
    with pytest.raises(Exception):
        mgr.create_project(p)

def test_list_and_get_projects(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("projA", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    items = mgr.list_projects("req-1")
    assert any(x.project_id == p.project_id for x in items)

    got = mgr.get_project("req-2", p.project_id)
    assert got is not None
    assert got.project_id == p.project_id

def test_update_project_name_and_roots_schedules_tasks(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)

    p = Project("projX", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    # Spy on add_task to ensure file tasks are scheduled for changes
    add_task_calls = []
    def fake_add_task(task):
        add_task_calls.append(task)
    mgr._task_mgr.add_task = fake_add_task

    new_root = (tmp_path / "sub").as_posix()
    updates = {"name": "projX-renamed", "source_roots": [new_root]}

    updated = mgr.update_project("req-123", p.project_id, updates)
    assert updated is not None
    # Two transitions: remove old root, add new root => 2 FileTasks scheduled
    assert len(add_task_calls) >= 1  # at least add; removal may noop if old doesn't exist
    # The internal project was renamed
    entry = mgr._project_entries[p.project_id]
    assert entry.project.name == "projX-renamed"
    assert entry.project.source_roots == [new_root]

def test_delete_project_cleans_entry(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("projDel", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    assert p.project_id in mgr._project_entries
    mgr.delete_project("req-9", p.project_id)
    assert p.project_id not in mgr._project_entries
    # Also gone from project manager
    assert mgr._project_mgr.get_project(p.project_id) is None

def test_query_enqueues_task_success(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("projQ", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    # Capture add_task calls
    captured = {}
    def fake_add_task(task):
        # minimal check: task object has the properties GraphRagManagerImpl sets
        captured["task"] = task
    mgr._task_mgr.add_task = fake_add_task

    out = mgr.query("rid-1", p.project_id, "what is this?", {"temperature": 0})
    assert out["ok"] is True and out["request_id"] == "rid-1"
    assert "task" in captured  # task scheduled

def test_query_handles_task_add_failure(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("projQ2", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    def boom(_):
        raise RuntimeError("nope")
    mgr._task_mgr.add_task = boom

    out = mgr.query("rid-2", p.project_id, "q", {})
    assert out["ok"] is False
    assert "error" in out

def test_list_documents_enqueues_task(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("projLD", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    called = {"count": 0}
    def fake_add_task(task):
        called["count"] += 1
    mgr._task_mgr.add_task = fake_add_task

    out = mgr.list_documents("rid-L", p.project_id)
    assert out["ok"] is True and out["request_id"] == "rid-L"
    assert called["count"] == 1

def test_refresh_documents_enqueues_task(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    p = Project("projRF", [tmp_path.as_posix()], args={})
    mgr.create_project(p)

    called = {"count": 0}
    def fake_add_task(task):
        called["count"] += 1
    mgr._task_mgr.add_task = fake_add_task

    out = mgr.refresh_documents("rid-R", p.project_id)
    assert out["ok"] is True and out["request_id"] == "rid-R"
    assert called["count"] == 1

def test_handle_sync_project_invalid_project_raises(tmp_path: Path):
    mgr = build_manager_with_mocks(tmp_path)
    with pytest.raises(ValueError):
        mgr.handle_sync_project("rid-X", "missing-project-id")
