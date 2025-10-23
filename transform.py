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
from typing import Iterable, List

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import umap

from database import (
    get_sql_conn,
    fetch_page_vectors,
    store_blob,
    store_json,
)

logger = logging.getLogger(__name__)


def _batch_iterator(
    sqlconn, columns: List[str], batch_size: int = 10_000
) -> Iterable[dict]:
    """Yield rows in batches to avoid loading the entire table into memory.

    ``columns`` are passed to ``fetch_page_vectors``. Each yielded dict contains
    the requested columns plus ``page_id``.
    """
    offset = 0
    while True:
        # SQLite does not support OFFSET in a straightforward way without
        # ordering, but for our use-case ordering is not required - we simply
        # fetch a limited number of rows each iteration.
        sql = (
            f"SELECT page_id, {', '.join(columns)} FROM page_vector "
            f"LIMIT {batch_size} OFFSET {offset}"
        )
        cursor = sqlconn.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            break
        for row in rows:
            result = {"page_id": row["page_id"]}
            for col in columns:
                result[col] = row[col]
            yield result
        offset += batch_size


def run_pca(sqlconn, target_dim: int = 100, batch_size: int = 10_000) -> None:
    """Fit Incremental PCA on all ``embedding_vector`` blobs and store the result.

    The function streams data in batches to keep memory usage low. After fitting,
    it performs a second pass to transform each vector and stores the reduced
    representation in ``page_vector.reduced_vector``.
    """
    logger.info("Starting Incremental PCA (target_dim=%s)", target_dim)

    # First pass - fit
    ipca = IncrementalPCA(n_components=target_dim)
    for batch in _batch_iterator(sqlconn, ["embedding_vector"], batch_size):
        vectors = np.frombuffer(batch["embedding_vector"], dtype=np.float32)
        # Original embeddings are 2048-dimensional; reshape accordingly.
        vectors = vectors.reshape(1, -1)  # each batch here is a single row
        ipca.partial_fit(vectors)

    # Second pass - transform and store
    for batch in _batch_iterator(sqlconn, ["embedding_vector"], batch_size):
        vec = np.frombuffer(batch["embedding_vector"], dtype=np.float32).reshape(1, -1)
        reduced = ipca.transform(vec).astype(np.float32).squeeze()
        store_blob(sqlconn, batch["page_id"], "reduced_vector", reduced)

    logger.info("PCA completed and reduced vectors stored.")


def run_kmeans(sqlconn, n_clusters: int = 100, batch_size: int = 10_000) -> None:
    """Cluster the PCA-reduced vectors and persist cluster assignments.

    A standard ``KMeans`` model (fullbatch) is used because the reduced data
    size (100 dimensions) comfortably fits in memory for the typical dataset.
    """
    logger.info("Fetching reduced vectors for K-Means (n_clusters=%s)", n_clusters)

    # Load all reduced vectors into memory - they are small enough.
    rows = []
    ids = []
    for batch in _batch_iterator(sqlconn, ["reduced_vector"], batch_size):
        ids.append(batch["page_id"])
        rows.append(np.frombuffer(batch["reduced_vector"], dtype=np.float32))
    if not rows:
        logger.warning("No reduced vectors found - aborting K-Means.")
        return
    X = np.vstack(rows)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(X)

    # Store cluster ids and initialise cluster_info entries.
    for page_id, cl_id in zip(ids, cluster_ids):
        # Direct integer update for cluster_id
        sqlconn.execute(
            "UPDATE page_vector SET cluster_id = ? WHERE page_id = ?",
            (int(cl_id), page_id),
        )
        # Ensure a row exists in cluster_info; centroid will be updated later.
        sqlconn.execute(
            "INSERT OR IGNORE INTO cluster_info (cluster_id) VALUES (?)",
            (int(cl_id),),
        )
    sqlconn.commit()
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
        embedding = reducer.fit_transform(vectors).astype(np.float32)
        # Store each 3-D vector as JSON.
        for pid, vec in zip(page_ids, embedding):
            store_json(sqlconn, pid, "three_d_vector", vec.tolist())
        # Update centroid in cluster_info.
        centroid = embedding.mean(axis=0).tolist()
        sqlconn.execute(
            "UPDATE cluster_info SET centroid_3d = ? WHERE cluster_id = ?",
            (json.dumps(centroid), cl_id),
        )
        processed += 1
    sqlconn.commit()
    logger.info("UMAP mapping completed for %s clusters.", processed)
    return processed
