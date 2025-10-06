import logging
import uuid
from typing import Any, Dict, List, Optional, LiteralString

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
            # 1) Create per-project database by name
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run(
                    "CREATE DATABASE $db IF NOT EXISTS",
                    db=project.name,
                )

            # After DB creation, ensure required fulltext indexes exist in the project's database
            try:
                with self.driver.session(database=project.name) as proj_sess:
                    index_statements = [
                        # Full-text over content
                        """
                        CREATE FULLTEXT INDEX document_content_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.content]
                        """,
                        # Full-text over calls (array of strings)
                        """
                        CREATE FULLTEXT INDEX document_calls_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.calls]
                        """,
                        # Full-text over file_path
                        """
                        CREATE FULLTEXT INDEX document_file_path_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.file_path]
                        """,
                        # Full-text over symbol_name (on Chunk nodes)
                        """
                        CREATE FULLTEXT INDEX document_symbol_name_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.symbol_name]
                        """,
                        # Full-text over symbol_scope (on Chunk nodes)
                        """
                        CREATE FULLTEXT INDEX document_symbol_scope_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.symbol_scope]
                        """,
                        # Create/ensure indexes for Document navigation fields
                        """
                        CREATE INDEX document_next_idx IF NOT EXISTS 
                        FOR (d:Document) ON (d.next)
                        """,
                        # Create/ensure indexes for Document navigation fields
                        """
                        CREATE INDEX document_previous_idx IF NOT EXISTS
                        FOR (d:Document) ON (d.previous)
                        """
                    ]
                    for stmt in index_statements:
                        try:
                            proj_sess.run(stmt)
                            logger.debug("Ensured index: %s", stmt.strip().splitlines()[0])
                        except Exception as e:
                            logger.warning("Failed creating index in project DB %s: %s", project.name, e)
            except Exception as e:
                logger.exception("Error ensuring full-text indexes for project %s: %s", project.name, e)
                # Non-fatal; continue so project is at least created

            # 2) Insert metadata into catalog
            with self.driver.session(database=self.CATALOG_DB) as cat_sess:
                cat_sess.run(
                    """
                    CREATE (p:Project {
                        project_id: $project_id,
                        name: $name,
                        source_roots: $source_roots
                    })
                    """,
                    project_id=project.project_id,
                    name=project.name,
                    source_roots=list(project.source_roots or [])
                )

            return {
                "project_id": project.project_id,
                "name": project.name,
                "source_roots": project.source_roots
            }
        except Exception as e:
            logger.exception("Error in create_project: %s", e)
            raise

    def get_project(self, project_id: str) -> Optional[Project]:
        logger.info("get_project called: id=%s", project_id)
        try:
            with self.driver.session(database=self.CATALOG_DB) as sess:
                rec = sess.run(
                    "MATCH (p:Project {project_id: $id}) RETURN p LIMIT 1", id=project_id
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
                            "source_roots": list(p.get("source_roots", []))
                    }))
                return items
        except Exception as e:
            logger.exception("Error in list_projects: %s", e)
            raise

    def update_project(
        self,
        project_id: str,
        args: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        logger.info("update_project called: id=%s", project_id)
        current = self.get_project(project_id)
        if current is None:
            raise ValueError(f"Project not found: {project_id}")

        if args is None or len(args) == 0:
            return current.__dict__

        try:
            # If name changes, we need to rename database: create new DB and drop old
            old_name = current.name
            new_name = args.get("name", old_name)
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
            for key, val in args.items():
                set_clauses.append(f"p.{key} = ${key}")
                params[key] = val

            set_fragment = ", ".join(set_clauses)
            cypher: LiteralString = f"""
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
                    "embedder_model_name": p["embedder_model_name"],
                    "llm_model_name": p["llm_model_name"],
                    "query_temperature": p["query_temperature"]
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

    def stop(self) -> None:
        pass
