from dataclasses import dataclass
import json
from typing import Optional


@dataclass
class Chunk:
    chunk_name: str
    namespace: str
    chunk_archive_path: Optional[str] = None
    chunk_extracted_path: Optional[str] = None
    downloaded_at: Optional[str] = None
    unpacked_at: Optional[str] = None


@dataclass
class Page:
    page_id: int
    title: str
    chunk_name: str
    url: str
    extracted_at: Optional[str] = None
    abstract: Optional[str] = None

    def to_json(self, pretty_print: bool = False):
        return json.dumps(self.__dict__, indent=2) if pretty_print else json.dumps(self.__dict__)


@dataclass
class PageVectors:
    page_id: int
    embedding_vector: Optional[bytes] = None  # Stored as a blob
    reduced_vector: Optional[bytes] = None  # Stored as a blob
    cluster_id: Optional[int] = None
    three_d_vector: Optional[str] = None  # Stored as JSON string "[x, y, z]"


@dataclass
class ClusterInfo:
    cluster_id: int
    namespace: str
    centroid_3d: Optional[str] = None  # Stored as JSON string
    cluster_name: Optional[str] = None
    cluster_description: Optional[str] = None


Vector3D = tuple[float, float, float]
