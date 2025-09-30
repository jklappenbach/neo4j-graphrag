from typing import List, Dict, Any, Optional

from pydantic import BaseModel

class ProjectCreate(BaseModel):
    name: str
    source_roots: List[str]
    args: Dict[str, Any] = {}

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    source_roots: Optional[List[str]] = None
    args: Optional[Dict[str, Any]] = None

class SyncRequest(BaseModel):
    force: bool = False
