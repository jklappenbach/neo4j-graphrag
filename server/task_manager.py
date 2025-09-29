import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Callable, Any, Dict, Optional, Tuple, List

import neo4j
from neo4j import Record
from watchdog.events import FileSystemEvent

from server.code_change_handler import EventType
from server.task_dao import TaskDAO

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enumeration of possible processing statuses."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

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

class Task(ABC):
    _request_id: str
    _task_status: TaskStatus
    _status_tracker: Optional[TaskManager]
    _created_at: float = field(default_factory=time.time)
    _started_at: Optional[float] = None
    _completed_at: Optional[float] = None
    _result: Optional[Dict[str, Any]] = None
    _error: Optional[str] = None
    _is_finished: Optional[bool] = None
    _is_cancelled: Optional[bool] = None
    _total_time: Optional[float] = None
    _execution_time: Optional[float] = None

    def __init__(self, request_id: str, status_tracker: Optional[TaskManager] = None) -> None:
        """Initialize a new task for processing."""
        self._request_id = request_id
        self._status_tracker = status_tracker
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
        
    @classmethod
    def from_record(cls, record: Record) -> 'Task':
        """Create a Task instance from a database record."""
        # Create instance with minimal args first
        instance = cls.__new__(cls)  # Create without calling __init__
        
        # Initialize from record
        instance._request_id = record.get("request_id")
        instance._task_status = TaskStatus(record.get("status"))
        instance._status_tracker = None  # Not available when reconstructing from DB
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
    def get_record_cypher(self) -> bool:
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

class FileTask(Task):
    def __init__(self, request_id: str, file_system_event: FileSystemEvent,
                 event_type: EventType,
                 handler: Callable[[FileSystemEvent], bool],
                 status_tracker: TaskManager) -> None:
        super().__init__(request_id, status_tracker)
        self._event = file_system_event
        self._event_type = event_type
        self._handler = handler

    @classmethod
    def from_record(cls, record: Record) -> 'FileTask':
        """Create a FileTask instance from a database record."""
        # Create base task from record
        instance = cls.__new__(cls)
        Task.__init__(instance, record.get("request_id"))
        
        # Set task fields from record
        instance._request_id = record.get("request_id")
        instance._task_status = TaskStatus(record.get("status"))
        instance._status_tracker = None
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
        
        # Set FileTask-specific fields from record
        instance._event_type = EventType(record.get("event_type"))
        instance._src_path = record.get("src_path")
        instance._dest_path = record.get("dest_path")
        instance._is_directory = record.get("is_directory")
        instance._handler = None  # Cannot reconstruct handler from DB
        instance._event = None    # Cannot reconstruct event from DB
        
        return instance

    def get_task_type(self) -> str:
        return "FileTask"

    def execute(self):
        if self._is_cancelled:
            return
        try:
            if self._status_tracker:
                self._status_tracker.start_task(self._request_id)
            if self._handler and self._event:
                self._handler(self._event)
            if self._status_tracker:
                self._status_tracker.complete_task(self._request_id, {})
        except Exception as e:
            if self._status_tracker:
                self._status_tracker.fail_task(self._request_id, str(e))

    def get_record_cypher(self) -> Tuple[str, Dict[str, Any]]:
        """Create a FileEventStatus record in Neo4j."""
        query = """
        CREATE (f:ProcessingStatus:FileTask {
            request_id: $request_id,
            task_type: $task_type,
            event_type: $event_type,
            src_path: $src_path,
            dest_path: $dest_path,
            is_directory: $is_directory,
            status: $status,
            created_at: $created_at,
            started_at: $started_at,
            completed_at: $completed_at,
            result: $result,
            error: $error
        })
        RETURN f.request_id as request_id
        """

        parameters = {
            "request_id": self._request_id,
            "task_type": self.get_task_type(),
            "event_type": self._event_type.value,
            "src_path": self._event.src_path if self._event else self._src_path,
            "dest_path": self._event.dest_path if self._event else self._dest_path,
            "is_directory": self._event.is_directory if self._event else self._is_directory,
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }

        return query, parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for serialization."""
        result = super().to_dict()
        result["event_type"] = self._event_type.value if hasattr(self, '_event_type') else None
        result["src_path"] = self._event.src_path if self._event else getattr(self, '_src_path', None)
        result["dest_path"] = self._event.dest_path if self._event else getattr(self, '_dest_path', None)
        result["is_directory"] = self._event.is_directory if self._event else getattr(self, '_is_directory', None)
        return result

class QueryTask(Task):
    def __init__(self, request_id: str, query: str,
                 handler: Callable[[str, str], dict[str, Any]],
                 status_tracker: TaskManager) -> None:
        super().__init__(request_id, status_tracker)
        self._query = query
        self._handler = handler

    @classmethod
    def from_record(cls, record: Record) -> 'QueryTask':
        """Create a QueryTask instance from a database record."""
        instance = cls.__new__(cls)
        Task.__init__(instance, record.get("request_id"))
        
        # Set task fields from record
        instance._request_id = record.get("request_id")
        instance._task_status = TaskStatus(record.get("status"))
        instance._status_tracker = None
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
        
        # Set QueryTask-specific fields
        instance._query = record.get("query")
        instance._handler = None  # Cannot reconstruct handler from DB
        
        return instance

    def get_task_type(self) -> str:
        return "QueryTask"

    def execute(self):
        if self._is_cancelled:
            return
        try:
            self._status_tracker.start_task(self._request_id)
            query_result = self._handler(self._request_id, self._query)
            if self._status_tracker:
                self._status_tracker.complete_task(self._request_id, query_result)
        except Exception as e:
            self._status_tracker.fail_task(self._request_id, str(e))

    def get_record_cypher(self) -> Tuple[str, Dict[str, Any]]:
        query = """
        CREATE (q:ProcessingStatus:QueryTask {
            request_id: $request_id,
            task_type: $task_type,
            query: $query,
            status: $status,
            created_at: $created_at,
            started_at: $started_at,
            completed_at: $completed_at,
            result: $result,
            error: $error
        })
        RETURN q.request_id as request_id
        """

        parameters = {
            "request_id": self._request_id,
            "query": self._query,
            "task_type": self.get_task_type(),
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }
        return query, parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for serialization."""
        result = super().to_dict()
        result["query"] = self._query
        return result

class RefreshTask(Task):
    def __init__(self, request_id: str,
                 handler: Callable[['RefreshTask'], bool],
                 status_tracker: TaskManager) -> None:
        super().__init__(request_id, status_tracker)
        self._handler = handler

    @classmethod
    def from_record(cls, record: Record) -> 'RefreshTask':
        """Create a RefreshTask instance from a database record."""
        instance = cls.__new__(cls)
        Task.__init__(instance, record.get("request_id"))
        
        # Set task fields from record
        instance._request_id = record.get("request_id")
        instance._task_status = TaskStatus(record.get("status"))
        instance._status_tracker = None
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
        
        # Set RefreshTask-specific fields
        instance._handler = None  # Cannot reconstruct handler from DB
        
        return instance

    def execute(self):
        if self._is_cancelled:
            return
        try:
            if self._status_tracker:
                self._status_tracker.start_task(self._request_id)
            if self._handler:
                self._handler(self)
            if self._status_tracker:
                self._status_tracker.complete_task(self._request_id, {})
        except Exception as e:
            if self._status_tracker:
                self._status_tracker.fail_task(self._request_id, str(e))

    def get_task_type(self) -> str:
        return "RefreshTask"

    def get_record_cypher(self) -> Tuple[str, Dict[str, Any]]:
        query = """
        CREATE (q:ProcessingStatus:RefreshTask {
            request_id: $request_id,
            task_type: $task_type,
            status: $status,
            created_at: $created_at,
            started_at: $started_at,
            completed_at: $completed_at,
            result: $result,
            error: $error
        })
        RETURN q.request_id as request_id
        """

        parameters = {
            "request_id": self._request_id,
            "task_type": self.get_task_type(),
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }
        return query, parameters

class ListDocumentsTask(Task):
    def __init__(self, request_id: str,
                 handler: Callable[[str], Dict[str, Any]],
                 status_tracker: TaskManager) -> None:
        super().__init__(request_id, status_tracker)
        self._handler = handler

    @classmethod
    def from_record(cls, record: Record) -> 'ListDocumentsTask':
        """Create a ListDocumentsTask instance from a database record."""
        instance = cls.__new__(cls)
        Task.__init__(instance, record.get("request_id"))
        
        # Set task fields from record
        instance._request_id = record.get("request_id")
        instance._task_status = TaskStatus(record.get("status"))
        instance._status_tracker = None
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
        
        # Set ListDocumentsTask-specific fields
        instance._handler = None  # Cannot reconstruct handler from DB
        
        return instance

    def execute(self):
        if self._is_cancelled:
            return
        try:
            if self._status_tracker:
                self._status_tracker.start_task(self._request_id)
            if self._handler:
                result = self._handler(self._request_id)
                if self._status_tracker:
                    self._status_tracker.complete_task(self._request_id, result)
        except Exception as e:
            if self._status_tracker:
                self._status_tracker.fail_task(self._request_id, str(e))

    def get_task_type(self) -> str:
        return "ListDocumentsTask"

    def get_record_cypher(self) -> Tuple[str, Dict[str, Any]]:
        query = """
        CREATE (q:ProcessingStatus:ListDocumentsTask {
            request_id: $request_id,
            task_type: $task_type,
            status: $status,
            created_at: $created_at,
            started_at: $started_at,
            completed_at: $completed_at,
            result: $result,
            error: $error
        })
        RETURN q.request_id as request_id
        """

        parameters = {
            "request_id": self._request_id,
            "task_type": self.get_task_type(),
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }
        return query, parameters

class TaskFactory:
    @staticmethod
    def create_from_record(record: Record) -> Task:
        task_type = record.get("task_type")
        match task_type:
            case "ListDocumentsTask":
                return ListDocumentsTask.from_record(record)
            case "RefreshTask":
                return RefreshTask.from_record(record)
            case "FileTask":
                return FileTask.from_record(record)
            case "QueryTask":
                return QueryTask.from_record(record)
            case _:
                raise ValueError(f"Unknown task type: {task_type}")
# EventExecutor Code
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Dict, Any, Awaitable

class TaskManagerImpl(TaskManager):
    def __init__(self, driver: neo4j.Driver,
                 websocket_notifier: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None):
        self._records: Dict[str, Task] = {}
        self._dao = TaskDAO(driver)
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.RLock()
        self._websocket_notifier = websocket_notifier

    def add_task(self, task: Task) -> None:
        try:
            with self._lock:
                self._thread_pool.submit(self._execute_task_wrapper, task)
                task.enqueue()
                self._dao.create_task_record(task)
                self._records[task._request_id] = task
                if self._websocket_notifier:
                    asyncio.create_task(self._websocket_notifier(task._request_id, {
                        "type": "task_enqueued",
                        "task_type": task.get_task_type(),
                        "success": True,
                        "task_info": task.to_dict()
                    }))
        except Exception as e:
            logger.exception("Error adding task: %s", str(e))
            raise e

    def start_task(self, task: Task) -> None:
        try:
            with self._lock:
                task.start_processing()
                self._dao.update_task_record(task)
                
                # Send WebSocket notification
                if self._websocket_notifier:
                    asyncio.create_task(self._websocket_notifier(task._request_id, {
                        "type": "task_started",
                        "task_type": task.get_task_type(),
                        "success": True,
                        "task_info": task.to_dict()
                    }))
        except Exception as e:
            logger.exception("Error starting task: %s", str(e))
            return False

    def complete_task(self, task: Task, result: Dict[str, Any]) -> None:
        try:
            with self._lock:
                task.complete_with_result(result)
                self._dao.update_task_record(task)
                
                # Send WebSocket notification
                if self._websocket_notifier:
                    asyncio.create_task(self._websocket_notifier(task._request_id, {
                        "type": "task_completed",
                        "task_type": task.get_task_type(),
                        "success": True,
                        "result": result,
                        "task_info": task.to_dict()
                    }))
        except Exception as e:
            logger.exception("Error completing task: %s", str(e))
            return False

    def fail_task(self, task: Task, error: str) -> None:
        try:
            with self._lock:
                task.fail_with_error(error)
                self._dao.update_task_record(task)
                
                # Send WebSocket notification
                if self._websocket_notifier:
                    asyncio.create_task(self._websocket_notifier(task._request_id, {
                        "type": "task_failed",
                        "task_type": task.get_task_type(),
                        "success": False,
                        "error": error,
                        "task_info": task.to_dict()
                    }))
        except Exception as e:
            logger.exception("Error failing task: %s", str(e))
            return False

    def cancel_task(self, request_id: str) -> None:
        try:
            with self._lock:
                task = self._records.get(request_id)
                if task:
                    task.cancel()
                    self._dao.update_task_record(task)
                    
                    # Send WebSocket notification
                    if self._websocket_notifier:
                        asyncio.create_task(self._websocket_notifier(request_id, {
                            "type": "task_cancelled",
                            "task_type": task.get_task_type(),
                            "success": True,
                            "task_info": task.to_dict()
                        }))
                return False
        except Exception as e:
            logger.exception("Error canceling task: %s", str(e))
            return False

    def get_task(self, request_id: str) -> Optional[Task]:
        """Get a task by request ID."""
        with self._lock:
            return self._records.get(request_id)

    def list_active_tasks(self) -> List[Task]:
        """Get all active (non-finished) tasks."""
        with self._lock:
            return [task for task in self._records.values() if not task.is_finished]