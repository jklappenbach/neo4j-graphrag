from typing import List, Dict, Any, Optional

from pydantic import BaseModel

class ProjectCreate(BaseModel):
    name: str
    source_roots: List[str]
    args: Dict[str, Any] = {}

class SyncRequest(BaseModel):
    force: bool = False
