import logging
import uuid
from enum import Enum
from pathlib import Path
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from typing import TYPE_CHECKING, List

from server.task_manager import TaskManagerImpl, FileTask

if TYPE_CHECKING:
    from graph_rag_manager import GraphRagManager

logger = logging.getLogger(__name__)

class EventType(Enum):
    FILE_CHANGED = "file_changed"
    PATH_DELETED = "file_deleted"
    PATH_MOVED = "path_moved"
    PATH_CREATED = "path_created"

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, mgr: 'GraphRagManager', event_executor: TaskManagerImpl) -> None:
        self._mgr = mgr
        self._event_executor = event_executor

    def on_modified(self, event: FileSystemEvent) -> None:
        self._event_executor.add_task(FileTask(uuid.uuid4(), event, self._handle_modified))

    def on_moved(self, event: FileSystemEvent) -> None:
        self._event_executor.add_task(FileTask(uuid.uuid4(), event, self._handle_moved))

    def on_created(self, event: FileSystemEvent) -> None:
        self._event_executor.add_task(FileTask(uuid.uuid4(), event, self._handle_created))

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._event_executor.add_task(FileTask(uuid.uuid4(), event, self._handle_deleted))

    def _handle_modified(self, event: FileSystemEvent) -> bool:
        """Direct handler for modified events (bypasses queuing)."""
        logger.info("Processing queued modified event: src_path=%s is_directory=%s", 
                   event.src_path, event.is_directory)
        
        if event.is_directory:
            self._handle_moved(event)
        else:
            self._mgr._update_path(Path(event.src_path))

    def _handle_moved(self, event: FileSystemEvent) -> None:
        """
        IF the event is sa directory, we've changed its name.  Treat as a move.
        Otherwise, delete the file, and re-ingest
        :param event:
        :return:
        """
        logger.info("Entering CodeChangeHandler.on_modified event_type=%s src_path=%s is_directory=%s",
                    event.event_type, event.src_path, event.is_directory)

        if event.is_directory:
            self._mgr._delete_path(Path(event.src_path))
        else:
            self._mgr._update_path(Path(event.src_path))

    def _handle_created(self, event: FileSystemEvent) -> None:
        """Direct handler for created events (bypasses queuing)."""
        logger.info("Processing queued created event: src_path=%s is_directory=%s", 
                   event.src_path, event.is_directory)
        
        path = Path(event.src_path)
        self._mgr._add_path(path)

    def _handle_deleted(self, event: FileSystemEvent) -> None:
        """Direct handler for deleted events (bypasses queuing)."""
        logger.info("Processing queued deleted event: src_path=%s is_directory=%s", 
                   event.src_path, event.is_directory)
        
        self._mgr._delete_path(Path(event.src_path))
