from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray


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
    namespace: str
    page_id: int
    title: str
    chunk_name: str
    url: str
    extracted_at: Optional[str] = None
    abstract: Optional[str] = None


@dataclass
class ClusterTreeNode:
    namespace: str
    node_id: int
    depth: int
    parent_id: Optional[int] = None
    child_count: Optional[int] = 0
    doc_count: Optional[int] = None
    centroid: Optional[NDArray] = None
    top_terms: Optional[list[str]] = None
    sample_doc_ids: Optional[list[int]] = None
    first_label: Optional[str] = None
    final_label: Optional[str] = None


@dataclass
class PageContent:
    page_id: int
    title: str
    abstract: str


@dataclass
class ClusterNodeTopics:
    node_id: int
    depth: int
    parent_id: Optional[int]
    first_label: Optional[str]
    final_label: Optional[str]
    is_leaf: bool
