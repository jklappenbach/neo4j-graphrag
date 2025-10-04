import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Callable, List
from typing import Tuple, LiteralString

import neo4j
from neo4j import Record
from watchdog.events import FileSystemEvent

from server.server_defines import Task, TaskManager, FileEventType, WebSocketManager
from server.task_dao import TaskDAO

logger = logging.getLogger(__name__)


class FileTask(Task):
    def __init__(self, request_id: str, project_id: str, src_path: str, dest_path: str, is_directory: bool,
                 event_type: FileEventType,
                 handler: Callable[[str, str, bool], None],
                 task_mgr: TaskManager) -> None:
        super().__init__(request_id, project_id, task_mgr)
        self._event_type = event_type
        self._handler = handler
        self._src_path = src_path
        self._dest_path = dest_path
        self._is_directory = is_directory

    @classmethod
    def from_record(cls, record: Record) -> Task:
        """Create a FileTask instance from a database record."""
        # Create base task from record
        instance = cls.from_record(record)

        # Set FileTask-specific fields from record
        instance._event_type = FileEventType(record.get("event_type"))
        instance._src_path = record.get("src_path")
        instance._dest_path = record.get("dest_path")
        instance._is_directory = record.get("is_directory")
        instance._handler = None  # Cannot reconstruct handler from DB

        return instance

    def get_task_type(self) -> str:
        return "FileTask"

    def execute(self):
        if self._is_cancelled:
            return
        try:
            if self._task_mgr:
                self._task_mgr.start_task(self._request_id)
            if self._handler:
                self._handler(self._event_type, self._src_path, self._dest_path, self._is_directory)
            if self._task_mgr:
                self._task_mgr.complete_task(self._request_id, {})
        except Exception as e:
            if self._task_mgr:
                self._task_mgr.fail_task(self._request_id, str(e))

    def get_create_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        """Create a FileEventStatus record in Neo4j."""
        query: LiteralString = """
        CREATE (f:ProcessingStatus:FileTask {
            request_id: $request_id,
            project_id: $project_id,
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

        parameters: Dict[str, Any] = {
            "request_id": self._request_id,
            "project_id": self._project_id,
            "task_type": self.get_task_type(),
            "event_type": self._event_type.value,
            "src_path": self._src_path,
            "dest_path": self._dest_path,
            "is_directory": self._is_directory,
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }

        return query, parameters

    def get_update_cypher(self) -> tuple[str, Dict[str, Any]]:
        query = """
        MATCH (f:ProcessingStatus:FileTask {request_id: $request_id})
        SET f.task_type = $task_type,
            f.project_id = $project_id,
            f.event_type = $event_type,
            f.src_path = $src_path,
            f.dest_path = $dest_path,
            f.is_directory = $is_directory,
            f.status = $status,
            f.created_at = $created_at,
            f.started_at = $started_at,
            f.completed_at = $completed_at,
            f.result = $result,
            f.error = $error
        RETURN f.request_id as request_id
        """

        parameters = {
            "request_id": self._request_id,
            "project_id": self._project_id,
            "task_type": self.get_task_type(),
            "event_type": self._event_type.value,
            "src_path": self._src_path,
            "dest_path": self._dest_path,
            "is_directory": self._is_directory,
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
        result["event_type"] = self._event_type
        result["src_path"] = self._src_path
        result["dest_path"] = self._dest_path
        result["is_directory"] = self._is_directory
        return result


class QueryTask(Task):
    def __init__(self, request_id: str, project_id: str, query: str, args: Dict[str, Any],
                 handler: Callable[[str, str, str, Dict[str, Any]], Dict[str, Any]],
                 task_mgr: TaskManager) -> None:
        super().__init__(request_id, project_id, task_mgr)
        self._query = query
        self._handler = handler
        self._temperature = args.get('temperature', 1)
        self._embedding_model_name = args.get('embedding_model_name', 'default')
        self._llm_model_name = args.get('llm_model_name', 'default')
        self._query_prefix = args.get('query_prefix', 'default')

    @classmethod
    def from_record(cls, record: Record) -> 'QueryTask':
        """Create a QueryTask instance from a database record."""
        instance = cls.from_record(record)

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
            self._task_mgr.start_task(self._request_id)
            query_result = self._handler(self._request_id, self._query)
            if self._task_mgr:
                self._task_mgr.complete_task(self._request_id, query_result)
        except Exception as e:
            self._task_mgr.fail_task(self._request_id, str(e))

    def get_create_cypher(self) -> tuple[str, Dict[str, str | float | None]]:
        query = """
        CREATE (q:ProcessingStatus:QueryTask {
            request_id: $request_id,
            project_id: $project_id,
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
            "project_id": self._project_id,
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

    def get_update_cypher(self) -> tuple[str, Dict[str, str | float | None]]:
        query = """
        MATCH (f:ProcessingStatus:QueryTask {request_id: $request_id})
        SET f.task_type = $task_type,
            f.project_id = $project_id,
            f.query = $query,
            f.status = $status,
            f.created_at = $created_at,
            f.started_at = $started_at,
            f.completed_at = $completed_at,
            f.result = $result,
            f.error = $error
        RETURN f.request_id as request_id
        """

        parameters = {
            "request_id": self._request_id,
            "project_id": self._project_id,
            "task_type": self.get_task_type(),
            "query": self._query,
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
                 project_id: str,
                 handler: Callable[[str], None],
                 task_mgr: TaskManager) -> None:
        super().__init__(request_id, project_id, task_mgr)
        self._handler = handler

    @classmethod
    def from_record(cls, record: Record) -> 'RefreshTask':
        """Create a RefreshTask instance from a database record."""
        instance = cls.from_record(record)

        # TODO Set RefreshTask-specific fields
        instance._handler = None  # Cannot reconstruct handler from DB

        return instance

    def execute(self):
        if self._is_cancelled:
            return
        try:
            if self._task_mgr:
                self._task_mgr.start_task(self._request_id)
            if self._handler:
                self._handler(self._request_id)
            if self._task_mgr:
                self._task_mgr.complete_task(self._request_id, {})
        except Exception as e:
            if self._task_mgr:
                self._task_mgr.fail_task(self._request_id, str(e))

    def get_task_type(self) -> str:
        return "RefreshTask"

    def get_create_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        query: LiteralString = """
        CREATE (q:ProcessingStatus:RefreshTask {
            request_id: $request_id,
            project_id: $project_id,
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
            "project_id": self._project_id,
            "task_type": self.get_task_type(),
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }
        return query, parameters

    def get_update_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        query: LiteralString = """
        MATCH (f:ProcessingStatus:RefreshTask {request_id: $request_id})
        SET f.task_type = $task_type,
            f.query = $query,
            f.status = $status,
            f.created_at = $created_at,
            f.started_at = $started_at,
            f.completed_at = $completed_at,
            f.result = $result,
            f.error = $error
        RETURN f.request_id as request_id
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
    def __init__(self, request_id: str, project_id: str,
                 handler: Callable[[str], Dict[str, Any]],
                 task_mgr: TaskManager) -> None:
        super().__init__(request_id, project_id, task_mgr)
        self._handler = handler

    @classmethod
    def from_record(cls, record: Record) -> 'ListDocumentsTask':
        """Create a ListDocumentsTask instance from a database record."""
        instance = cls.from_record(record)

        # TODO Set ListDocumentsTask-specific fields
        instance._handler = None  # Cannot reconstruct handler from DB

        return instance

    def execute(self):
        if self._is_cancelled:
            return
        try:
            if self._task_mgr:
                self._task_mgr.start_task(self._request_id)
            if self._handler:
                result = self._handler(self._request_id)
                if self._task_mgr:
                    self._task_mgr.complete_task(self._request_id, result)
        except Exception as e:
            if self._task_mgr:
                self._task_mgr.fail_task(self._request_id, str(e))

    def get_task_type(self) -> str:
        return "ListDocumentsTask"

    def get_create_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        query: LiteralString = """
        CREATE (q:ProcessingStatus:ListDocumentsTask {
            request_id: $request_id,
            project_id: $project_id,
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

        parameters: Dict[str, Any] = {
            "request_id": self._request_id,
            "project_id": self._project_id,
            "task_type": self.get_task_type(),
            "status": self._task_status.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "result": json.dumps(self._result) if self._result else None,
            "error": self._error
        }
        return query, parameters

    def get_update_cypher(self) -> Tuple[LiteralString, Dict[str, Any]]:
        query: LiteralString = """
        MATCH (f:ProcessingStatus:ListTask {request_id: $request_id})
        SET f.task_type = $task_type,
            f.query = $query,
            f.status = $status,
            f.created_at = $created_at,
            f.started_at = $started_at,
            f.completed_at = $completed_at,
            f.result = $result,
            f.error = $error
        RETURN f.request_id as request_id
        """

        parameters: Dict[str, Any] = {
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

class TaskManagerImpl(TaskManager):

    def __init__(self, driver: neo4j.Driver,
                 websocket_manager: WebSocketManager):
        self._records: Dict[str, Task] = {}
        self._dao = TaskDAO(driver)
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.RLock()
        self._websocket_manager = websocket_manager

    def _notify(self, request_id: str, payload: Dict[str, Any]) -> None:
        """
        Send a notification to all connected WebSocket clients using WebSocketManager.
        Ensures proper behavior whether we're in an async or sync context.
        """
        # Include request_id in the payload
        message = dict(payload)
        message["request_id"] = request_id

        async def _send_all():
            try:
                await self._websocket_manager.send_message_all(message)
            except Exception as e:
                logger.exception("WebSocket broadcast failed for request_id=%s: %s", request_id, str(e))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            loop.create_task(_send_all())
        else:
            asyncio.run(_send_all())

    def add_task(self, task: Task) -> None:
        try:
            with self._lock:
                # submit expects a callable, not the result -> pass function, not function()
                self._thread_pool.submit(task.execute)
                task.enqueue()
                query, parameters = task.get_create_cypher()
                self._dao.create_task_record(query, parameters)
                self._records[task.request_id] = task
                self._notify(task.request_id, {
                        "type": "task_enqueued",
                        "task_type": task.get_task_type(),
                        "success": True,
                        "task_info": task.to_dict(),
                    })
        except Exception as e:
            logger.exception("Error adding task: %s", str(e))
            raise e

    def start_task(self, task: Task) -> None:
        try:
            with self._lock:
                task.start_processing()
                query, parameters = task.get_update_cypher()
                self._dao.update_task_record(query, parameters)
                self._notify(task.request_id, {
                        "type": "task_started",
                        "task_type": task.get_task_type(),
                        "success": True,
                        "task_info": task.to_dict()
                    })
        except Exception as e:
            logger.exception("Error starting task: %s", str(e))

    def complete_task(self, task: Task, result: Dict[str, Any]) -> None:
        try:
            with self._lock:
                task.complete_with_result(result)
                query, parameters = task.get_update_cypher()
                self._dao.update_task_record(query, parameters)
                self._notify(task.request_id, {
                        "type": "task_completed",
                        "task_type": task.get_task_type(),
                        "success": True,
                        "result": result,
                        "task_info": task.to_dict()
                    })
        except Exception as e:
            logger.exception("Error completing task: %s", str(e))

    def fail_task(self, task: Task, error: str) -> None:
        try:
            with self._lock:
                task.fail_with_error(error)
                query, properties = task.get_update_cypher()
                self._dao.update_task_record(query, properties)
                self._notify(task.request_id, {
                        "type": "task_failed",
                        "task_type": task.get_task_type(),
                        "success": False,
                        "error": error,
                        "task_info": task.to_dict()
                    })
        except Exception as e:
            logger.exception("Error failing task: %s", str(e))

    def cancel_task(self, request_id: str) -> None:
        try:
            with self._lock:
                task = self._records.get(request_id)
                if task:
                    task.cancel()
                    query, properties = task.get_update_cypher()
                    self._dao.update_task_record(query, properties)
                    self._notify(task.request_id, {
                            "type": "task_cancelled",
                            "task_type": task.get_task_type(),
                            "success": True,
                            "task_info": task.to_dict()
                        })
        except Exception as e:
            logger.exception("Error canceling task: %s", str(e))

    def get_task(self, request_id: str) -> Optional[Task]:
        """Get a task by request ID."""
        with self._lock:
            return self._records.get(request_id)

    def list_active_tasks(self, request_id: str) -> List[Task]:
        """Get all active (non-finished) tasks."""
        logger.info("Listing active tasks for request_id: %s", request_id)
        with self._lock:
            return [task for task in self._records.values() if not task.is_finished]

    def stop(self) -> None:
        self._thread_pool.shutdown(wait=True)