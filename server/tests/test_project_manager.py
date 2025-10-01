"""Unit tests for ProjectManagerImpl component."""
import pytest
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
from server.project_manager import ProjectManagerImpl
from server.server_defines import Project


class MockNeo4jNode:
    """Mock Neo4j node."""
    def __init__(self, props: Dict[str, Any]):
        self._props = props
    
    def __getitem__(self, key: str):
        return self._props.get(key)
    
    def get(self, key: str, default=None):
        return self._props.get(key, default)


class MockResult:
    """Mock Neo4j result."""
    def __init__(self, records: Optional[List[Dict[str, Any]]] = None, single_record: Optional[Dict[str, Any]] = None):
        self._records = records or []
        self._single = {"p": MockNeo4jNode(single_record)} if single_record else None
    
    def single(self):
        return self._single
    
    def __iter__(self):
        for rec in self._records:
            yield {"p": MockNeo4jNode(rec)}


class MockSession:
    """Mock Neo4j session."""
    def __init__(self, database: str = "system"):
        self.database = database
        self.run_calls = []
        self._projects_db = {}
    
    def run(self, query: str, **params):
        self.run_calls.append((query, params))
        
        # System database operations
        if self.database == "system":
            return MockResult()
        
        # Catalog database operations
        if self.database == "graph-rag":
            if "CREATE (p:Project" in query:
                proj_id = params.get("project_id", str(uuid.uuid4()))
                self._projects_db[proj_id] = {
                    "project_id": proj_id,
                    "name": params.get("name"),
                    "source_roots": params.get("source_roots", []),
                    "llm_model_name": "default",
                    "embedder_model_name": "default",
                    "query_temperature": 1.0
                }
                return MockResult()
            elif "MATCH (p:Project {id:" in query:
                proj_id = params.get("id")
                proj = self._projects_db.get(proj_id)
                return MockResult(single_record=proj)
            elif "MATCH (p:Project) RETURN p" in query:
                return MockResult(records=list(self._projects_db.values()))
            elif "SET" in query and "MATCH (p:Project" in query:
                proj_id = params.get("project_id")
                if proj_id in self._projects_db:
                    for key, val in params.items():
                        if key != "project_id":
                            self._projects_db[proj_id][key] = val
                    return MockResult(single_record=self._projects_db[proj_id])
                return MockResult()
            elif "DETACH DELETE p" in query:
                proj_id = params.get("project_id")
                if proj_id in self._projects_db:
                    del self._projects_db[proj_id]
                return MockResult()
        
        return MockResult()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class MockDriver:
    """Mock Neo4j driver."""
    def __init__(self):
        self._sessions = {}
    
    def session(self, database: str = "system"):
        if database not in self._sessions:
            self._sessions[database] = MockSession(database=database)
        return self._sessions[database]


@pytest.fixture
def mock_driver():
    """Fixture providing a mock Neo4j driver."""
    return MockDriver()


@pytest.fixture
def project_manager(mock_driver):
    """Fixture providing ProjectManagerImpl with mocked dependencies."""
    with patch.object(ProjectManagerImpl, '_ensure_catalog'):
        return ProjectManagerImpl(mock_driver)


# Valid test cases
def test_create_project_success(project_manager):
    """Test successfully creating a project."""
    project = Project(name="TestProject", source_roots=["/src"])
    
    result = project_manager.create_project(project)
    
    assert "project_id" in result
    assert result["name"] == "TestProject"
    assert result["source_roots"] == ["/src"]

def test_get_project_success(project_manager):
    """Test successfully retrieving a project."""
    project = Project(name="GetProject", source_roots=["/src"])
    created = project_manager.create_project(project)
    proj_id = created["project_id"]
    
    # Mock the get to return something
    with patch.object(project_manager.driver.session("graph-rag"), 'run') as mock_run:
        mock_run.return_value = MockResult(single_record={
            "project_id": proj_id,
            "name": "GetProject",
            "source_roots": ["/src"],
            "llm_model_name": "default",
            "embedder_model_name": "default",
            "query_temperature": 1.0
        })
        
        retrieved = project_manager.get_project(proj_id)
        
        assert retrieved is not None
        assert retrieved.name == "GetProject"


def test_list_projects_success(project_manager):
    """Test successfully listing projects."""
    project1 = Project(name="Project1", source_roots=["/src1"])
    project2 = Project(name="Project2", source_roots=["/src2"])
    
    project_manager.create_project(project1)
    project_manager.create_project(project2)
    
    with patch.object(project_manager.driver.session("graph-rag"), 'run') as mock_run:
        mock_run.return_value = MockResult(records=[
            {"project_id": "id1", "name": "Project1", "source_roots": ["/src1"], "args": {}},
            {"project_id": "id2", "name": "Project2", "source_roots": ["/src2"], "args": {}}
        ])
        
        projects = project_manager.list_projects()
        
        assert len(projects) == 2
        names = [p.name for p in projects]
        assert "Project1" in names
        assert "Project2" in names


def test_update_project_success(project_manager):
    """Test successfully updating a project."""
    project = Project(name="UpdateProject", source_roots=["/src"])
    created = project_manager.create_project(project)
    proj_id = created["project_id"]
    
    with patch.object(project_manager, 'get_project') as mock_get:
        mock_get.return_value = {
            "project_id": proj_id,
            "name": "UpdateProject",
            "source_roots": ["/src"],
            "llm_model_name": "default",
            "embedder_model_name": "default",
            "query_temperature": 1.0
        }
        
        with patch.object(project_manager.driver.session("graph-rag"), 'run') as mock_run:
            mock_run.return_value = MockResult(single_record={
                "project_id": proj_id,
                "name": "UpdatedProject",
                "source_roots": ["/newsrc"],
                "llm_model_name": "default",
                "embedder_model_name": "default",
                "query_temperature": 1.0
            })
            
            updated = project_manager.update_project(
                proj_id,
                args = { 'name': 'UpdatedProject',
                'source_roots': ['/newsrc'],
                'llm_model_name': 'default',
                'embedder_model_name': 'default',
                'query_temperature': 1.0 }
            )
            
            assert updated["name"] == "UpdatedProject"
            assert updated["source_roots"] == ["/newsrc"]

def test_delete_project_success(project_manager):
    """Test successfully deleting a project."""
    project = Project(name="DeleteProject", source_roots=["/src"])
    created = project_manager.create_project(project)
    proj_id = created["project_id"]
    
    with patch.object(project_manager, 'get_project') as mock_get:
        mock_get.return_value = {
            "project_id": proj_id,
            "name": "DeleteProject",
            "source_roots": ["/src"],
            "llm_model_name": "default",
            "embedder_model_name": "default",
            "query_temperature": 1.0

        }
        
        result = project_manager.delete_project(proj_id)
        
        assert result["deleted"] is True
        assert result["project_id"] == proj_id


# Invalid test cases
def test_get_project_not_found(project_manager):
    """Test retrieving a non-existent project."""
    with patch.object(project_manager.driver.session("graph-rag"), 'run') as mock_run:
        mock_run.return_value = MockResult(single_record=None)
        
        result = project_manager.get_project("nonexistent-id")
        
        assert result is None


def test_delete_project_not_found(project_manager):
    """Test deleting a non-existent project."""
    with patch.object(project_manager, 'get_project', return_value=None):
        result = project_manager.delete_project("nonexistent-id")
        
        assert result["deleted"] is False
        assert result["reason"] == "not_found"


def test_update_project_not_found(project_manager):
    """Test updating a non-existent project."""
    with patch.object(project_manager, 'get_project', return_value=None):
        with pytest.raises(ValueError, match="Project not found"):
            project_manager.update_project("nonexistent-id", args={'name': 'NewName'})


def test_create_project_with_database_error(project_manager):
    """Test creating project when database error occurs."""
    project = Project(name="ErrorProject", source_roots=["/src"])
    
    with patch.object(project_manager.driver.session("system"), 'run', side_effect=Exception("DB Error")):
        with pytest.raises(Exception, match="DB Error"):
            project_manager.create_project(project)


def test_list_projects_with_database_error(project_manager):
    """Test listing projects when database error occurs."""
    with patch.object(project_manager.driver.session("graph-rag"), 'run', side_effect=Exception("DB Error")):
        with pytest.raises(Exception, match="DB Error"):
            project_manager.list_projects()


def test_update_project_with_no_changes(project_manager):
    """Test updating project with no actual changes."""
    project = Project(name="NoChangeProject", source_roots=["/src"])
    created = project_manager.create_project(project)
    proj_id = created["project_id"]
    
    current_data = {
        "project_id": proj_id,
        "name": "NoChangeProject",
        "source_roots": ["/src"],
        "llm_model_name": "default",
        "embedder_model_name": "default",
        "query_temperature": 1.0
    }
    
    with patch.object(project_manager, 'get_project', return_value=current_data):
        # No updates provided
        result = project_manager.update_project(proj_id, { 'source_roots': ["/src"]})
        
        # Should return current data unchanged
        assert result == current_data
