import logging
import uuid

from watchdog.events import FileSystemEventHandler, FileSystemEvent

from server.server_defines import TaskManager, GraphRagManager, FileEventType
from server.task_manager import FileTask

logger = logging.getLogger(__name__)

class ProjectFileSystemEventHandler(FileSystemEventHandler):
    def __init__(self, project_id: str, task_mgr: TaskManager, graph_rag_mgr: GraphRagManager):
        self._task_mgr = task_mgr
        self._graphrag_mgr = graph_rag_mgr
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
                                         handler=self._graphrag_mgr.handle_update_path,
                                         listener=self._task_mgr))

    def on_moved(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(request_id=str(uuid.uuid4()),
                                         project_id=self._project_id,
                                         event_type=FileEventType.PATH_MOVED,
                                         src_path=event.src_path,
                                         dest_path=event.dest_path,
                                         is_directory=event.is_directory,
                                         handler=self._graphrag_mgr.handle_update_path,
                                         listener=self._task_mgr))

    def on_created(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(request_id=str(uuid.uuid4()),
                                         project_id=self._project_id,
                                         event_type=FileEventType.PATH_CREATED,
                                         src_path=event.src_path,
                                         dest_path=event.dest_path,
                                         is_directory=event.is_directory,
                                         handler=self._graphrag_mgr.handle_add_path,
                                         listener=self._task_mgr))

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._task_mgr.add_task(FileTask(request_id=str(uuid.uuid4()),
                                         project_id=self._project_id,
                                         event_type=FileEventType.PATH_DELETED,
                                         src_path=event.src_path,
                                         dest_path=event.dest_path,
                                         is_directory=event.is_directory,
                                         handler=self._graphrag_mgr.handle_delete_path,
                                         listener=self._task_mgr))
