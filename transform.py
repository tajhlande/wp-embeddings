"""Dimensionality reduction and clustering utilities.

This module implements the three major steps described in
`.roo/dimensionality_processing_plan.md`:

1. **PCA** - reduce the original 2048-dimensional embedding vectors to a
   configurable target dimension (default 100).
2. **K-Means** - cluster the PCA-reduced vectors.
3. **UMAP** - map each cluster to a 3-D space for visualisation.

All heavy-lifting is delegated to scikit-learn and umap-learn. The functions
operate directly on the SQLite database via the helper utilities defined in
``database.py``.
"""

from __future__ import annotations

import json
import logging
logger = logging.getLogger(__name__)
import sqlite3
from typing import Iterable, List, Optional

import numpy as np

logger.info("Initializing scikit-learn...")
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans, MiniBatchKMeans
import umap.umap_ as umap

from database import (
    get_sql_conn,
    get_page_embeddings,
    get_page_reduced_vectors,
    update_cluster_centroid,
    update_reduced_vector_for_page,
    update_three_d_vector_for_page,
)
from progress_utils import ProgressTracker

logger = logging.getLogger(__name__)


def _batch_iterator(
    sqlconn: sqlite3.Connection, 
    namespace: str, 
    columns: List[str], 
    batch_size: int = 10_000
) -> Iterable[List[dict]]:
    """Yield batches of rows to avoid loading the entire table into memory.

    ``columns`` are passed to ``fetch_page_vectors``. Each yielded list contains
    batch_size dicts, each with the requested columns plus ``page_id``.
    """
    offset = 0
    while True:
        # SQLite does not support OFFSET in a straightforward way without
        # ordering, but for our use-case ordering is not required - we simply
        # fetch a limited number of rows each iteration.
        sql = (
            f"""
            SELECT page_vector.page_id, {', '.join(columns)} 
            FROM page_vector 
            INNER JOIN page_log ON page_vector.page_id = page_log.page_id
            INNER JOIN chunk_log ON page_log.chunk_name = chunk_log.chunk_name
            WHERE chunk_log.namespace = :namespace
            {''.join([f"AND {column} IS NOT NULL " for column in columns])}
            ORDER BY chunk_log.chunk_name ASC, page_log.page_id ASC
            LIMIT {batch_size} OFFSET {offset};
            """
        )
        cursor = sqlconn.execute(sql, {'namespace': namespace})
        rows = cursor.fetchall()
        if not rows:
            break
        
        batch = []
        for row in rows:
            result = {"page_id": row["page_id"]}
            for col in columns:
                result[col] = row[col]
            batch.append(result)
        
        yield batch
        offset += batch_size


def run_pca(sqlconn: sqlite3.Connection, namespace: str, target_dim: int = 100, batch_size: int = 10_000, tracker: Optional[ProgressTracker] = None) -> tuple[int, int]:
    """
    Fit Incremental PCA on all ``embedding_vector`` blobs and store the result.

    The function streams data in batches to keep memory usage low. After fitting,
    it performs a second pass to transform each vector and stores the reduced
    representation in ``page_vector.reduced_vector``.

    Returns a tuple of the total batch count and the total vector count.
    """
    # logger.info("Starting Incremental PCA (target_dim=%s)", target_dim)

    # Ensure batch_size is at least target_dim for the first partial_fit call
    effective_batch_size = max(batch_size, target_dim)
    # logger.info("Using effective batch size: %s (requested: %s, target_dim: %s)",
    #             effective_batch_size, batch_size, target_dim)

    # First pass - fit
    ipca = IncrementalPCA(n_components=target_dim)
    batch_counter = 0
    total_vectors = 0

    for batch in _batch_iterator(sqlconn, namespace, ["embedding_vector"], effective_batch_size):
        # Stack all vectors in this batch into a single matrix
        batch_vectors = []
        for row in batch:
            vectors = np.frombuffer(row["embedding_vector"], dtype=np.float32)
            # Original embeddings are 2048-dimensional; reshape accordingly.
            vectors = vectors.reshape(1, -1)  # each row is a single vector
            batch_vectors.append(vectors)
        
        if batch_vectors:
            batch_matrix = np.vstack(batch_vectors)
            ipca.partial_fit(batch_matrix)
            tracker.update(1) if tracker else None
            batch_counter += 1
            total_vectors += len(batch)

    # Second pass - transform and store
    for batch in _batch_iterator(sqlconn, namespace, ["embedding_vector"], batch_size):
        # Process each vector in the batch
        for row in batch:
            vec = np.frombuffer(row["embedding_vector"], dtype=np.float32).reshape(1, -1)
            reduced = ipca.transform(vec).astype(np.float32).squeeze()
            update_reduced_vector_for_page(row["page_id"], reduced, sqlconn)
        tracker.update(1) if tracker else None

    # logger.info("PCA completed and reduced vectors stored.")
    return batch_counter, total_vectors


def run_kmeans(sqlconn: sqlite3.Connection,
               namespace: str,
               n_clusters: int = 100,
               batch_size: int = 10_000,
               tracker: Optional[ProgressTracker] = None) -> None:
    """Cluster the PCA-reduced vectors and persist cluster assignments.

    Args:
        sqlconn: SQLite database connection
        namespace: Namespace to process
        n_clusters: Number of clusters to create
        batch_size: Batch size for processing
        tracker: Optional progress tracker
        use_incremental: If True, use MiniBatchKMeans for incremental processing.
                        If False, use standard KMeans (loads all data into memory).
    """
    # logger.info("Fetching reduced vectors for K-Means (n_clusters=%s, incremental=%s)",
    #             n_clusters, use_incremental)

    # if use_incremental:
    # Use MiniBatchKMeans for incremental processing
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size)
    ids = []
    partial_fit_done = False
    
    # First pass: partial fit on all batches
    # logger.info("Performing partial fit on all batches...")
    for batch in _batch_iterator(sqlconn, namespace, ["reduced_vector"], batch_size):
        batch_vectors = []
        batch_ids = []
        for row in batch:
            batch_ids.append(row["page_id"])
            vectors = np.frombuffer(row["reduced_vector"], dtype=np.float32)
            batch_vectors.append(vectors)
        
        if batch_vectors:
            batch_matrix = np.vstack(batch_vectors)
            if not partial_fit_done:
                kmeans.partial_fit(batch_matrix)
                partial_fit_done = True
            else:
                kmeans.partial_fit(batch_matrix)
            
            ids.extend(batch_ids)
        tracker.update(1) if tracker else None
    
    if not ids:
        logger.warning("No reduced vectors found - aborting K-Means.")
        return
    
    # Second pass: predict on batches to get cluster assignments
    logger.info("Predicting cluster assignments...")
    cluster_ids = []
    for batch in _batch_iterator(sqlconn, namespace, ["reduced_vector"], batch_size):
        batch_vectors = []
        batch_ids = []
        for row in batch:
            batch_ids.append(row["page_id"])
            vectors = np.frombuffer(row["reduced_vector"], dtype=np.float32)
            batch_vectors.append(vectors)
        
        if batch_vectors:
            batch_matrix = np.vstack(batch_vectors)
            batch_cluster_ids = kmeans.predict(batch_matrix)
            cluster_ids.extend(batch_cluster_ids)
            ids.extend(batch_ids)
            tracker.update(1) if tracker else None

    # Store cluster ids and initialise cluster_info entries.
    for page_id, cl_id in zip(ids, cluster_ids):
        # Direct integer update for cluster_id
        sqlconn.execute(
            "UPDATE page_vector SET cluster_id = ? WHERE page_id = ?",
            (int(cl_id), page_id),
        )
        # Ensure a row exists in cluster_info; centroid will be updated later.
        sqlconn.execute(
            "INSERT OR IGNORE INTO cluster_info (cluster_id, namespace) VALUES (?, ?)",
            (int(cl_id), namespace),
        )
    
    # Save cluster centroids as BLOB
    logger.info("Saving cluster centroids...")
    for cluster_id in range(n_clusters):
        centroid = kmeans.cluster_centers_[cluster_id]
        update_cluster_centroid(cluster_id, namespace, centroid, sqlconn)
    logger.info("K-Means clustering completed and assignments stored.")


def run_umap_per_cluster(
    sqlconn,
    n_components: int = 3,
    limit: int = 1,
    batch_size: int = 10_000,
) -> int:
    """Apply UMAP within each cluster and store 3-D vectors.

    ``limit`` caps the number of clusters processed (useful for quick tests).
    Returns the number of clusters actually processed.
    """
    logger.info(
        "Running UMAP per cluster (n_components=%s, limit=%s)", n_components, limit
    )
    # Determine distinct cluster ids.
    cur = sqlconn.execute("SELECT DISTINCT cluster_id FROM page_vector WHERE cluster_id IS NOT NULL")
    cluster_ids = [row["cluster_id"] for row in cur.fetchall()][:limit]

    processed = 0
    for cl_id in cluster_ids:
        # Gather all reduced vectors for this cluster.
        cur = sqlconn.execute(
            "SELECT page_id, reduced_vector FROM page_vector WHERE cluster_id = ?",
            (cl_id,),
        )
        rows = cur.fetchall()
        if not rows:
            continue
        page_ids = [r["page_id"] for r in rows]
        vectors = np.vstack(
            [np.frombuffer(r["reduced_vector"], dtype=np.float32) for r in rows]
        )
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        three_space_vectors = reducer.fit_transform(vectors).astype(np.float32) # type: ignore
        # Store each 3-D vector as JSON.
        for pid, vec in zip(page_ids, three_space_vectors):
            update_three_d_vector_for_page(pid, vec.tolist(), sqlconn)
        # Update centroid in cluster_info.
        # centroid = three_space_vectors.mean(axis=0).tolist()
        # sqlconn.execute(
        #     "UPDATE cluster_info SET centroid_3d = ? WHERE cluster_id = ?",
        #     (json.dumps(centroid), cl_id),
        # )
        processed += 1
    sqlconn.commit()
    logger.info("UMAP mapping completed for %s clusters.", processed)
    return processed
