import logging
import uuid
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileSystemEvent

from server.server_defines import TaskManager, GraphRagManager, FileEventType
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
        self._task_mgr.add_task(FileTask(request_id=str(uuid.uuid4()),
                                         project_id=self._project_id,
                                         event_type=FileEventType.FILE_CHANGED,
                                         src_path=event.src_path,
                                         dest_path=event.dest_path,
                                         is_directory=event.is_directory,
                                         handler=self.handle_modified,
                                         task_mgr=self._task_mgr))

    def on_moved(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(str(uuid.uuid4()),
                                         self._project_id,
                                         event.src_path,
                                         event.dest_path,
                                         event.is_directory,
                                         FileEventType.PATH_MOVED,
                                         self.handle_moved,
                                         self._task_mgr))

    def on_created(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(str(uuid.uuid4()),
                                         self._project_id,
                                         event.src_path,
                                         event.dest_path,
                                         event.is_directory,
                                         FileEventType.PATH_CREATED,
                                         self.handle_created,
                                         self._task_mgr))

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(uuid.uuid4(), self._project_id, event, self.handle_deleted))

    def handle_modified(self, src_path: str, dest_path: str, is_directory: bool) -> None:
        """Direct handler for modified events (bypasses queuing)."""
        logger.info("Processing queued modified event: src_path=%s is_directory=%s",
                    src_path, is_directory)

        if is_directory:
            self.handle_moved(src_path, dest_path, is_directory)
        else:
            self._graph_rag_mgr.handle_update_path(self._project_id, Path(src_path))

    def handle_moved(self, src_path: str, dest_path: str, is_directory: bool) -> None:
        logger.info("Entering CodeChangeHandler.on_modified project_id=%s src_path=%s is_directory=%s",
                    self._project_id, src_path, is_directory)

        self._graph_rag_mgr.handle_delete_path(self._project_id, Path(src_path))
        self._graph_rag_mgr.handle_add_path(self._project_id, Path(dest_path))

    def handle_created(self, src_path: str, dest_path: str, is_directory: bool) -> None:
        """Direct handler for created events (bypasses queuing)."""
        logger.info("Processing queued created event: src_path=%s is_directory=%s",
                    src_path, is_directory)

        path = Path(src_path)
        self._graph_rag_mgr.handle_add_path(self._project_id, path)

    def handle_deleted(self, src_path: str, dest_path: str, is_directory: bool) -> None:
        """Direct handler for deleted events (bypasses queuing)."""
        logger.info("Processing queued deleted event: src_path=%s is_directory=%s",
                    src_path, is_directory)

        self._graph_rag_mgr.handle_delete_path(self._project_id, Path(src_path))
