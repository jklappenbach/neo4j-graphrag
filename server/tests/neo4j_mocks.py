from typing import Dict, Any, Optional, List


class MockNeo4jRecord:
    """Mock Neo4j record."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str):
        return self._data.get(key)

    def get(self, key: str, default=None):
        return self._data.get(key, default)


class MockResult:
    """Mock Neo4j result."""

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None, single_record: Optional[Dict[str, Any]] = None):
        self._records = [MockNeo4jRecord(r) for r in (records or [])]
        self._single = MockNeo4jRecord(single_record) if single_record else None

    def single(self):
        return self._single

    def __iter__(self):
        return iter(self._records)

    def fetch(self):
        return self._records

