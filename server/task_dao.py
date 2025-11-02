
"""Data Access Object for Task records in Neo4j."""
from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any, LiteralString

from typing import Any

# Optional neo4j imports for test environments without the package installed
try:  # pragma: no cover
    import neo4j  # type: ignore
    from neo4j import Record  # type: ignore
    from neo4j.exceptions import Neo4jError  # type: ignore
except Exception:  # pragma: no cover - provide fallbacks for tests
    neo4j = None  # type: ignore
    Record = Any  # type: ignore
    class Neo4jError(Exception):
        pass

from server.server_defines import TaskStatus

logger = logging.getLogger(__name__)

class TaskDAO:
    """Data Access Object for ProcessingStatus records in Neo4j."""
    
    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        # Create indexes on initialization
        self.create_indexes()

    def create_task_record(self, query: LiteralString, parameters: Dict[str, Any]) -> bool:
        """Create a Task record in Neo4j using the task's own Cypher query."""
        try:
            request_id = parameters.get("request_id")
            logger.info("Creating task record: %s", parameters.get("request_id"))
        
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()
                
                if record is not None:
                    logger.info("Successfully created task record: %s", request_id)
                    return True
                else:
                    logger.warning("Task record creation returned no result: %s", request_id)
                    return False
                    
        except Exception as e:
            logger.exception("Failed to create task record: %s", str(e))
            return False

    def update_task_record(self, query: LiteralString, parameters: Dict[str, Any]) -> bool:
        """Update an existing task record in Neo4j."""
        try:
            request_id = parameters.get("request_id")
            logger.info("Updating task record: %s", parameters.get("request_id"))

            with self.driver.session() as session:
                # Use a generic update query that works for all task types
                result = session.run(query, parameters)

                if len(result.fetch()) > 0:
                    logger.info("Successfully updated task record: %s", request_id)
                    return True
                else:
                    logger.warning("Task record update returned no result: %s", request_id)
                    return False

        except Neo4jError as e:
            logger.exception("Failed to update task record: %s", e)
            return False
    
    def get_task_record(self, request_id: str) -> Optional[Record]:
        """Get a task record by request_id."""
        logger.debug("Getting task record: %s", request_id)
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:ProcessingStatus {request_id: $request_id})
                RETURN p.request_id as request_id,
                       p.task_type as task_type,
                       p.status as status,
                       p.created_at as created_at,
                       p.started_at as started_at,
                       p.completed_at as completed_at,
                       p.result as result,
                       p.error as error,
                       p.query as query,
                       p.event_type as event_type,
                       p.src_path as src_path,
                       p.dest_path as dest_path,
                       p.is_directory as is_directory
                """
                
                result = session.run(query, {"request_id": request_id})
                return result.single()
                
        except Exception as e:
            logger.exception("Failed to get task record: %s", e)
            return None

    def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> List[Record]:
        """Get tasks by their status."""
        logger.debug("Getting tasks by status: %s", status.value)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:ProcessingStatus {status: $status})
                RETURN p.request_id as request_id,
                       p.task_type as task_type,
                       p.status as status,
                       p.created_at as created_at,
                       p.started_at as started_at,
                       p.completed_at as completed_at,
                       p.result as result,
                       p.error as error,
                       p.query as query,
                       p.event_type as event_type,
                       p.src_path as src_path,
                       p.dest_path as dest_path,
                       p.is_directory as is_directory
                ORDER BY p.created_at DESC
                LIMIT $limit
                """

                result = session.run(query, {"status": status.value, "limit": limit})
                return list(result)

        except Exception as e:
            # Accept any exception type in tests/mocks, not only Neo4jError
            logger.exception("Failed to get tasks by status: %s", e)
            return []

    def get_recent_tasks(self, limit: int = 50) -> List[Record]:
        """Get the most recent tasks."""
        logger.debug("Getting recent tasks, limit: %s", limit)
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:ProcessingStatus)
                RETURN p.request_id as request_id,
                       p.task_type as task_type,
                       p.status as status,
                       p.created_at as created_at,
                       p.started_at as started_at,
                       p.completed_at as completed_at,
                       p.result as result,
                       p.error as error,
                       p.query as query,
                       p.event_type as event_type,
                       p.src_path as src_path,
                       p.dest_path as dest_path,
                       p.is_directory as is_directory
                ORDER BY p.created_at DESC
                LIMIT $limit
                """
                
                result = session.run(query, {"limit": limit})
                return list(result)
                
        except Neo4jError as e:
            logger.exception("Failed to get recent tasks: %s", e)
            return []

    def delete_task_record(self, request_id: str) -> bool:
        """Delete a task record by request_id."""
        logger.info("Deleting task record: %s", request_id)
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:ProcessingStatus {request_id: $request_id})
                DELETE p
                RETURN count(p) as deleted_count
                """
                
                result = session.run(query, {"request_id": request_id})
                record = result.single()
                deleted_count = record["deleted_count"] if record else 0
                
                if deleted_count > 0:
                    logger.info("Successfully deleted task record: %s", request_id)
                    return True
                else:
                    logger.warning("No task record found to delete: %s", request_id)
                    return False
                
        except Neo4jError as e:
            logger.exception("Failed to delete task record: %s", e)
            return False

    def cleanup_old_tasks(self, older_than_timestamp: float) -> int:
        """Delete task records older than the specified timestamp."""
        logger.info("Cleaning up tasks older than: %s", older_than_timestamp)
        
        try:
            with self.driver.session() as session:
                query: LiteralString = """
                MATCH (p:ProcessingStatus)
                WHERE p.created_at < $timestamp
                DELETE p
                RETURN count(p) as deleted_count
                """
                
                result = session.run(query, {"timestamp": older_than_timestamp})
                record = result.single()
                deleted_count = record["deleted_count"] if record else 0
                
                logger.info("Cleaned up %s old task records", deleted_count)
                return deleted_count
                
        except Exception as e:
            logger.exception("Failed to cleanup old tasks: %s", e)
            return 0

    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about tasks in the database."""
        logger.debug("Getting task statistics")
        
        try:
            with self.driver.session() as session:
                query: LiteralString = """
                MATCH (p:ProcessingStatus)
                RETURN p.status as status, 
                       p.task_type as task_type,
                       count(*) as count
                """
                
                result = session.run(query)
                
                stats = {
                    "total": 0,
                    "by_status": {},
                    "by_type": {}
                }
                
                for record in result:
                    status = record["status"]
                    task_type = record["task_type"]
                    count = record["count"]
                    
                    stats["total"] += count
                    
                    if status not in stats["by_status"]:
                        stats["by_status"][status] = 0
                    stats["by_status"][status] += count
                    
                    if task_type not in stats["by_type"]:
                        stats["by_type"][task_type] = 0
                    stats["by_type"][task_type] += count
                
                return stats
                
        except Exception as e:
            logger.exception("Failed to get task stats: %s", e)
            return {"total": 0, "by_status": {}, "by_type": {}}
    
    def create_indexes(self) -> bool:
        """Create indexes for better query performance."""
        logger.info("Creating indexes for ProcessingStatus nodes")
        
        try:
            with self.driver.session() as session:
                indexes: List[LiteralString] = [
                    "CREATE INDEX processing_status_request_id IF NOT EXISTS FOR (p:ProcessingStatus) ON (p.request_id)",
                    "CREATE INDEX processing_status_status IF NOT EXISTS FOR (p:ProcessingStatus) ON (p.status)",
                    "CREATE INDEX processing_status_type IF NOT EXISTS FOR (p:ProcessingStatus) ON (p.task_type)",
                    "CREATE INDEX processing_status_created_at IF NOT EXISTS FOR (p:ProcessingStatus) ON (p.created_at)"
                ]
                
                for index_query in indexes:
                    try:
                        session.run(index_query)
                        logger.debug("Created index: %s", index_query)
                    except Exception as e:
                        logger.warning("Failed to create index %s: %s", index_query, e)
                
                logger.info("Successfully processed ProcessingStatus indexes")
                return True
                
        except Neo4jError as e:
            logger.exception("Failed to create ProcessingStatus indexes: %s", e)
            return False
