# Python
import types
import pytest

from server.project_manager import ProjectManagerImpl

class _SessionStub:
    def __init__(self, database):
        self.database = database
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    # Provide a run stub that returns an object with .single() and is iterable
    def run(self, *args, **kwargs):
        class _Res:
            def __iter__(self): return iter([])
            def single(self): return None
        return _Res()

class _DriverStub:
    def session(self, database=None):
        return _SessionStub(database)

@pytest.fixture(scope="session")
def neo4j_driver():
    return _DriverStub()

@pytest.fixture
def project_manager(neo4j_driver):
    return ProjectManagerImpl(neo4j_driver)
