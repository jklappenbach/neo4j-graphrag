import dataclasses
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, LiteralString, List

from neo4j import Record
from websocket import WebSocket

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enumeration of possible processing statuses."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FileEventType(Enum):
    FILE_CHANGED = "file_changed"
    PATH_DELETED = "file_deleted"
    PATH_MOVED = "path_moved"
    PATH_CREATED = "path_created"

class Project:
    """A project represents a collection of source roots, which are recursively scanned for documents."""
    def __init__(self, name: str, source_roots: List[str], args: Dict[str, Any] = {}):
        self._project_id = str(uuid.uuid4())
        self._name = name
        self._source_roots = source_roots
        self._embedder_model_name = args.get('embedder_model_name', 'default')
        self._llm_model_name = args.get('llm_model_name', 'default')
        self._query_temperature = args.get('query_temperature', 1.0)

    @classmethod
    def from_dict(cls, src: Dict[str, Any]) -> 'Project':
        instance = cls.__new__(cls)
        instance._project_id = src.get('project_id') or str(uuid.uuid4())
        instance._name = src.get('name', '')
        instance._source_roots = src.get('source_roots') or src.get('src_roots') or []
        # Model/config fields with defaults
        instance._embedder_model_name = src.get('embedder_model_name', 'default')
        instance._llm_model_name = src.get('llm_model_name', 'default')
        instance._query_temperature = src.get('query_temperature', 1.0)
        return instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def source_roots(self) -> List[str]:
        return self._source_roots

    @property
    def embedder_model_name(self) -> str:
        return self._embedder_model_name
    @property
    def llm_model_name(self) -> str:
        return self._llm_model_name

    @property
    def query_temperature(self) -> float:
        return self._query_temperature

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Project name must be a non-empty string")
        self._name = value.strip()

    @source_roots.setter
    def source_roots(self, value: List[str]) -> None:
        if not isinstance(value, List):
            raise ValueError("Project source_roots must be a List[str]")
        self._source_roots = value

    @embedder_model_name.setter
    def embedder_model_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Project embedder_model_name must be a str")
        self._embedder_model_name = value

    @llm_model_name.setter
    def llm_model_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Project llm_model_name must be a str")
        self._llm_model_name = value

    @query_temperature.setter
    def query_temperature(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("Project query_temperature must be a float")
        self._query_temperature = value

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class TaskManager(ABC):
    """Abstract base class for status tracking implementations."""

    @abstractmethod
    def start_task(self, task) -> bool:
        """Mark a request as started."""
        pass

    @abstractmethod
    def complete_task(self, task, result: Dict[str, Any]) -> bool:
        """Mark a request as completed with result."""
        pass

    @abstractmethod
    def fail_task(self, task, error: str) -> bool:
        """Mark a request as failed with error."""
        pass

    @abstractmethod
    def add_task(self, task: 'Task'):
        pass

    @abstractmethod
    def list_active_tasks(self, request_id: str) -> List['Task']:
        pass

    @abstractmethod
    def cancel_task(self, request_id: str):
        pass

class Task(ABC):
    _request_id: str
    _project_id: str
    _task_status: TaskStatus
    _task_mgr: Optional[TaskManager]
    _created_at: float = field(default_factory=time.time)
    _started_at: Optional[float] = None
    _completed_at: Optional[float] = None
    _result: Optional[Dict[str, Any]] = None
    _error: Optional[str] = None
    _is_finished: Optional[bool] = None
    _is_cancelled: Optional[bool] = None
    _total_time: Optional[float] = None
    _execution_time: Optional[float] = None

    def __init__(self, request_id: str, project_id: str, task_mgr: Optional[TaskManager] = None) -> None:
        """Initialize a new task for processing."""
        self._request_id = request_id
        self._project_id = project_id
        self._task_mgr = task_mgr
        self._task_status = TaskStatus.QUEUED
        self._created_at = time.time()
        self._started_at = None
        self._completed_at = None
        self._result = None
        self._error = None
        self._is_finished = False
        self._is_cancelled = False
        self._total_time = None
        self._execution_time = None

    @property
    def request_id(self) -> str:
        return self._request_id

    @classmethod
    def from_record(cls, record: Record) -> Task:
        """Create a Task instance from a database record."""
        # Create instance with minimal args first
        instance = cls.__new__(cls)  # Create without calling __init__

        # Initialize from record
        instance._request_id = record.get("request_id")
        instance._project_id = record.get("project_id")
        instance._task_status = TaskStatus(record.get("status"))
        instance._task_mgr = None  # Not available when reconstructing from DB
        instance._created_at = record.get("created_at")
        instance._started_at = record.get("started_at")
        instance._completed_at = record.get("completed_at")
        instance._result = json.loads(record.get("result")) if record.get("result") else None
        instance._error = record.get("error")
        instance._is_finished = instance._task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        instance._is_cancelled = instance._task_status == TaskStatus.CANCELLED
        instance._total_time = time.time() - instance._created_at if instance._created_at else None
        instance._execution_time = (
                instance._completed_at - instance._started_at) if instance._completed_at and instance._started_at else None

        return instance

    @abstractmethod
    def get_create_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        pass

    def get_update_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        pass

    def enqueue(self) -> None:
        self._task_status = TaskStatus.QUEUED
        logger.info("Task %s queued", self._request_id)

    def start_processing(self) -> None:
        """Mark the task as started."""
        self._task_status = TaskStatus.PROCESSING
        self._started_at = time.time()
        logger.info("Task %s started processing", self._request_id)

    def complete_with_result(self, result: Dict[str, Any]) -> None:
        """Mark the task as completed with a result."""
        self._task_status = TaskStatus.COMPLETED
        self._completed_at = time.time()
        self._result = result
        self._is_finished = True
        logger.info("Task %s completed successfully", self._request_id)

    def fail_with_error(self, error: str) -> None:
        """Mark the task as failed with an error."""
        self._task_status = TaskStatus.FAILED
        self._completed_at = time.time()
        self._error = error
        self._is_finished = True
        logger.error("Task %s failed: %s", self._request_id, error)

    def cancel(self) -> None:
        """Mark the task as cancelled."""
        self._task_status = TaskStatus.CANCELLED
        self._completed_at = time.time()
        self._is_finished = True
        self._is_cancelled = True
        logger.info("Task %s cancelled", self._request_id)

    @property
    def is_finished(self) -> bool:
        """Check if the task has finished (completed, failed, or cancelled)."""
        return self._task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]

    @property
    def execution_time(self) -> Optional[float]:
        """Get the execution time in seconds, if available."""
        if self._started_at is None:
            return None

        end_time = self._completed_at if self._completed_at else time.time()
        return end_time - self._started_at

    @property
    def total_time(self) -> float:
        """Get the total time from creation to completion (or current time)."""
        end_time = self._completed_at if self._completed_at else time.time()
        return end_time - self._created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for serialization."""
        return {
            "request_id": self._request_id,
            "project_id": self._project_id,
            "task_type": self.get_task_type(),
            "status": self._task_status.value,
            "created_at": self._created_at,
            "created_at_iso": datetime.fromtimestamp(self._created_at).isoformat(),
            "started_at": self._started_at,
            "started_at_iso": datetime.fromtimestamp(self._started_at).isoformat() if self._started_at else None,
            "completed_at": self._completed_at,
            "completed_at_iso": datetime.fromtimestamp(self._completed_at).isoformat() if self._completed_at else None,
            "execution_time": self.execution_time,
            "total_time": self.total_time,
            "is_finished": self._is_finished,
            "result": self._result,
            "error": self._error
        }

class GraphRagManager(ABC):
    """
    Graph-based RAG manager using Haystack with code-aware ingestion.
    """

    SUPPORTED_EXTS = {".html", ".js", ".css", ".py", ".json"}

    # ---------------------
    # Basic API
    # ---------------------
    @abstractmethod
    def list_documents(self, request_id: str, project_id: str) -> List[str]:
        pass

    @abstractmethod
    def refresh_documents(self, request_id: str, project_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_project(self, project: Project) -> None:
        pass

    @abstractmethod
    def list_projects(self, request_id: str):
        pass

    @abstractmethod
    def get_project(self, request_id: str, project_id: str):
        pass

    @abstractmethod
    def update_project(self, request_id: str, project_id: str, args: Dict[str, Any]):
        pass

    @abstractmethod
    def delete_project(self, request_id: str, project_id: str) -> None:
        pass

    @abstractmethod
    def handle_sync_project(self, project_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def query(self, request_id: str, project_id: str, query_str: str, args: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def cancel_query(self, request_id: str):
        pass

    @abstractmethod
    def handle_add_path(self, request_id: str, project_id: str, src_path_str: str, dest_path_str: str) -> None:
        pass

    @abstractmethod
    def handle_update_path(self, request_id: str, project_id: str, src_path_str: str, dest_path_str: str) -> None:
        pass

    @abstractmethod
    def handle_delete_path(self, request_id: str, project_id: str, src_path_str: str, dest_path_str: str) -> None:
        pass

    @abstractmethod
    def handle_list_documents(self, project_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def handle_query(self, request_id: str, project_id: str, query: str, args: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass


class ProjectManager(ABC):
    """Abstract interface for CRUD operations on Project instances backed by Neo4j."""

    @abstractmethod
    def create_project(self, project: Project) -> Dict[str, Any]:
        """Create a new project record and associated Neo4j database."""
        raise NotImplementedError

    @abstractmethod
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a project by ID."""
        raise NotImplementedError

    @abstractmethod
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        raise NotImplementedError

    @abstractmethod
    def update_project(
        self,
        project_id: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update fields of a project."""
        raise NotImplementedError

    @abstractmethod
    def delete_project(self, project_id: str) -> Dict[str, Any]:
        """Delete a project and drop its Neo4j database."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        pass

class WebSocketManager(ABC):
    """Manages WebSocket connections and routing of TaskManager notifications to clients."""
    _task_mgr: TaskManager

    @abstractmethod
    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        pass

    @abstractmethod
    def disconnect(self, connection_id: str) -> None:
        pass

    @abstractmethod
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    async def send_message_all(self, message) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass
