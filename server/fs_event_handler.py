import logging
import uuid
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileSystemEvent

from server.server_defines import Project, TaskManager, GraphRagManager
from server.task_manager import FileTask

logger = logging.getLogger(__name__)

class ProjectFileSystemEventHandler(FileSystemEventHandler):
    def __init__(self, project_id: str, task_mgr: TaskManager, graph_rag_mgr: GraphRagManager):
        self._task_mgr = task_mgr
        self._graph_rag_mgr = graph_rag_mgr
        self._project_id = project_id

    # ---------------------
    # Watchdog Methods
    # ---------------------
    def on_modified(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(uuid.uuid4(), self._project_id, event, self.handle_modified))

    def on_moved(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(uuid.uuid4(), self._project_id, event, self.handle_moved))

    def on_created(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(uuid.uuid4(), self._project_id, event, self.handle_created))

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(uuid.uuid4(), self._project_id, event, self.handle_deleted))

    def handle_modified(self, event: FileSystemEvent) -> None:
        """Direct handler for modified events (bypasses queuing)."""
        logger.info("Processing queued modified event: src_path=%s is_directory=%s",
                    event.src_path, event.is_directory)

        if event.is_directory:
            self.handle_moved(event)
        else:
            self._graph_rag_mgr.handle_update_path(self._project_id, Path(event.src_path))

    def handle_moved(self, event: FileSystemEvent) -> None:
        """
        IF the event is sa directory, we've changed its name.  Treat as a move.
        Otherwise, delete the file, and re-ingest
        :param event:
        :return:
        """
        logger.info("Entering CodeChangeHandler.on_modified project_id=%s event_type=%s src_path=%s is_directory=%s",
                    self._project_id, event.event_type, event.src_path, event.is_directory)

        if event.is_directory:
            self._graph_rag_mgr.handle_delete_path(self._project_id, Path(event.src_path))
        else:
            self._graph_rag_mgr.handle_update_path(self._project_id, Path(event.src_path))

    def handle_created(self, event: FileSystemEvent) -> None:
        """Direct handler for created events (bypasses queuing)."""
        logger.info("Processing queued created event: src_path=%s is_directory=%s",
                    event.src_path, event.is_directory)

        path = Path(event.src_path)
        self._graph_rag_mgr.handle_add_path(self._project_id, path)

    def handle_deleted(self, event: FileSystemEvent) -> None:
        """Direct handler for deleted events (bypasses queuing)."""
        logger.info("Processing queued deleted event: src_path=%s is_directory=%s",
                    event.src_path, event.is_directory)

        self._graph_rag_mgr.handle_delete_path(self._project_id, Path(event.src_path))
