import logging
import uuid
from typing import Any, Dict, List, Optional

from neo4j import Driver

from server.server_defines import ProjectManager, Project

logger = logging.getLogger(__name__)

class ProjectManagerImpl(ProjectManager):
    """
    Neo4j-backed implementation.

    - Uses the system database 'system' to manage databases (create/drop).
    - Stores project metadata in a central catalog database 'projects_catalog'
      in a node labeled Project with properties:
        id (string, uuid), name (string), source_roots (list<string>), args (map<string,string>)
    - The per-project database is created/dropped with the project's name.
    """

    CATALOG_DB = "graph-rag"

    def __init__(self, driver: Driver):
        self.driver = driver
        # Ensure catalog DB and indexes exist
        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        logger.info("Ensuring catalog database and schema")
        try:
            # Ensure catalog database exists
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run(
                    "CREATE DATABASE $db IF NOT EXISTS",
                    db=self.CATALOG_DB)

            # Ensure unique constraint on Project.id
            with self.driver.session(database=self.CATALOG_DB) as cat_sess:
                cat_sess.run(
                    "CREATE CONSTRAINT project_id_unique IF NOT EXISTS "
                    "FOR (p:Project) REQUIRE p.id IS UNIQUE")
        except Exception as e:
            logger.exception("Failed to ensure catalog: %s", e)
            raise

    def create_project(self, project: Project) -> Dict[str, Any]:
        logger.info("create_project called: name=%s", project.name)
        try:
            project_id = str(uuid.uuid4())

            # 1) Create per-project database by name
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run(
                    "CREATE DATABASE $db IF NOT EXISTS",
                    db=project.name,
                )

            # 2) Insert metadata into catalog
            with self.driver.session(database=self.CATALOG_DB) as cat_sess:
                cat_sess.run(
                    """
                    CREATE (p:Project {
                        project_id: $project_id,
                        name: $name,
                        source_roots: $source_roots,
                        args: $args
                    })
                    """,
                    project_id=project_id,
                    name=project.name,
                    source_roots=list(project.source_roots or []),
                    args=dict(project.args or {}),
                )

            return {
                "project_id": project_id,
                "name": project.name,
                "source_roots": project.source_roots,
                "args": project.args,
                "database": project.name,
            }
        except Exception as e:
            logger.exception("Error in create_project: %s", e)
            raise

    def get_project(self, project_id: str) -> Optional[Project]:
        logger.info("get_project called: id=%s", project_id)
        try:
            with self.driver.session(database=self.CATALOG_DB) as sess:
                rec = sess.run(
                    "MATCH (p:Project {id: $id}) RETURN p LIMIT 1", id=project_id
                ).single()
                if not rec:
                    return None
                p = rec["p"]
                return Project.from_dict({
                    "project_id": p["project_id"],
                    "name": p["name"],
                    "source_roots": list(p.get("source_roots", [])),
                    "args": dict(p.get("args", {})),
                    "database": p["name"],
                })
        except Exception as e:
            logger.exception("Error in get_project: %s", e)
            raise

    def list_projects(self) -> List[Project]:
        logger.info("list_projects called")
        try:
            with self.driver.session(database=self.CATALOG_DB) as sess:
                res = sess.run("MATCH (p:Project) RETURN p ORDER BY p.name ASC")
                items: List[Project] = []
                for rec in res:
                    p = rec["p"]
                    items.append(Project.from_dict({
                            "project_id": p["project_id"],
                            "name": p["name"],
                            "source_roots": list(p.get("source_roots", [])),
                            "args": dict(p.get("args", {})),
                            "database": p["name"] }))
                return items
        except Exception as e:
            logger.exception("Error in list_projects: %s", e)
            raise

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        source_roots: Optional[List[str]] = None,
        args: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        logger.info("update_project called: id=%s", project_id)
        try:
            # Fetch current
            current = self.get_project(project_id)
            if current is None:
                raise ValueError(f"Project not found: {project_id}")

            updates = {}
            if name is not None:
                updates["name"] = name
            if source_roots is not None:
                updates["source_roots"] = list(source_roots)
            if args is not None:
                updates["args"] = dict(args)

            if not updates:
                return current

            # If name changes, we need to rename database: create new DB and drop old
            old_name = current["name"]
            new_name = updates.get("name", old_name)
            if new_name != old_name:
                with self.driver.session(database="system") as sys_sess:
                    # Create new database if not exists
                    sys_sess.run("CREATE DATABASE $db IF NOT EXISTS", db=new_name)
                    # Optionally, data migration could happen here if needed.
                    # Drop old database
                    sys_sess.run("DROP DATABASE $db IF EXISTS", db=old_name)

            # Apply metadata updates
            set_clauses = []
            params: Dict[str, Any] = {"project_id": project_id}
            for key, val in updates.items():
                set_clauses.append(f"p.{key} = ${key}")
                params[key] = val

            set_fragment = ", ".join(set_clauses)
            cypher = f"""
                MATCH (p:Project {{project_id: $project_id}})
                SET {set_fragment}
                RETURN p
            """

            with self.driver.session(database=self.CATALOG_DB) as sess:
                rec = sess.run(cypher, **params).single()
                p = rec["p"]
                return {
                    "project_id": p["project_id"],
                    "name": p["name"],
                    "source_roots": list(p.get("source_roots", [])),
                    "args": dict(p.get("args", {})),
                    "database": p["name"],
                }
        except Exception as e:
            logger.exception("Error in update_project: %s", e)
            raise

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        logger.info("delete_project called: id=%s", project_id)
        try:
            # Find project to know its database name
            proj = self.get_project(project_id)
            if proj is None:
                return {"deleted": False, "reason": "not_found"}

            db_name = proj["name"]

            # Delete metadata
            with self.driver.session(database=self.CATALOG_DB) as sess:
                sess.run("MATCH (p:Project {project_id: $project_id}) DETACH DELETE p", project_id=project_id)

            # Drop database
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run("DROP DATABASE $db IF EXISTS", db=db_name)

            return {"deleted": True, "project_id": project_id, "database": db_name}
        except Exception as e:
            logger.exception("Error in delete_project: %s", e)
            raise