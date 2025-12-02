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

import logging
import math
from typing import Iterable, Optional

import sqlite3
import numpy as np
from sklearn.metrics import silhouette_score
from numpy.typing import NDArray

from classes import ClusterTreeNode
from database import (
    get_clusters_needing_projection,
    get_reduced_vectors_for_cluster,
    update_cluster_tree_assignments,
    update_reduced_vectors_in_batch,
    update_three_d_vector_for_page,
    insert_cluster_tree_node,
    update_cluster_tree_child_count,
    get_cluster_tree_max_node_id,
    get_page_reduced_vectors,
    numpy_to_bytes,
)
from progress_utils import ProgressTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
numba_logger = logging.getLogger("numba.core")
numba_logger.setLevel(logging.WARNING)

# we are logging here because these import statements are sooooo slooooow
logger.info("Initializing scikit-learn...")
from sklearn.decomposition import IncrementalPCA  # noqa E402
from sklearn.cluster import MiniBatchKMeans  # noqa E402
import umap.umap_ as umap  # noqa E402


logger = logging.getLogger(__name__)


# def _batch_iterator(
#     sqlconn: sqlite3.Connection,
#     namespace: str,
#     columns: List[str],
#     batch_size: int = 10_000,
# ) -> Iterable[List[dict]]:
#     """Yield batches of rows to avoid loading the entire table into memory.

#     ``columns`` are passed to ``fetch_page_vectors``. Each yielded list contains
#     batch_size dicts, each with the requested columns plus ``page_id``.
#     """
#     offset = 0
#     while True:
#         # SQLite does not support OFFSET in a straightforward way without
#         # ordering, but for our use-case ordering is not required - we simply
#         # fetch a limited number of rows each iteration.
#         sql = f"""
#             SELECT page_id, {', '.join(columns)}
#             FROM page_vector
#             WHERE namespace = :namespace
#             {''.join([f"AND {column} IS NOT NULL " for column in columns])}
#             ORDER BY page_id ASC
#             LIMIT {batch_size} OFFSET {offset};
#             """
#         logger.debug("SQL query: %s", sql)
#         cursor = sqlconn.execute(sql, {"namespace": namespace})
#         rows = cursor.fetchall()
#         if not rows:
#             break

#         batch = []
#         for row in rows:
#             result = {"page_id": row["page_id"]}
#             for col in columns:
#                 result[col] = row[col]
#             batch.append(result)

#         yield batch
#         offset += batch_size


def _batch_iterator(
    sqlconn: sqlite3.Connection,
    namespace: str,
    columns: list[str],
    batch_size: int = 10_000,
) -> Iterable[list[dict]]:
    """Yield batches of rows without re-querying or using OFFSET."""
    sql = f"""
        SELECT page_id, {', '.join(columns)}
        FROM page_vector
        WHERE namespace = :namespace
        {''.join([f"AND {column} IS NOT NULL " for column in columns])}
        ORDER BY page_id ASC;
    """
    logger.debug("SQL query: %s", sql)
    cursor = sqlconn.execute(sql, {"namespace": namespace})

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        batch = [
            {"page_id": row["page_id"], **{col: row[col] for col in columns}}
            for row in rows
        ]
        yield batch


def run_pca(
    sqlconn: sqlite3.Connection,
    namespace: str,
    target_dim: int = 100,
    batch_size: int = 10_000,
    tracker: Optional[ProgressTracker] = None,
) -> tuple[int, int]:
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

    logger.debug("First pass: stacking vectors in matrix")
    # for batch in _batch_iterator(
    #     sqlconn, namespace, ["embedding_vector"], effective_batch_size
    # ):
    #     # Stack all vectors in this batch into a single matrix
    #     batch_vectors = []
    #     for row in batch:
    #         vectors = np.frombuffer(row["embedding_vector"], dtype=np.float32)
    #         # Original embeddings are 2048-dimensional; reshape accordingly.
    #         vectors = vectors.reshape(1, -1)  # each row is a single vector
    #         batch_vectors.append(vectors)

    #     if batch_vectors:
    #         batch_matrix = np.vstack(batch_vectors)
    #         ipca.partial_fit(batch_matrix)
    #         tracker.update(1) if tracker else None
    #         batch_counter += 1
    #         total_vectors += len(batch)

    for batch in _batch_iterator(sqlconn, namespace, ["embedding_vector"], effective_batch_size):
        # Decode all embeddings into a single matrix efficiently
        batch_matrix = np.frombuffer(b"".join(row["embedding_vector"] for row in batch), dtype=np.float32)
        batch_matrix = batch_matrix.reshape(len(batch), 2048)

        ipca.partial_fit(batch_matrix)
        tracker.update(1) if tracker else None
        batch_counter += 1
        total_vectors += len(batch)

    # def _embed_to_reduced_bytes(embed: NDArray) -> bytes:
    #     vec = np.frombuffer(embed, dtype=np.float32).reshape(
    #         1, -1
    #     )
    #     return ipca.transform(vec).astype(np.float32).squeeze().tobytes()

    # # Second pass - transform and store
    # logger.debug("Second pass: transforming and storing vectors in matrix")
    # for batch in _batch_iterator(sqlconn, namespace, ["embedding_vector"], batch_size):
    #     # Process each vector in the batch
    #     logger.debug("Transforming vectors")
    #     vectors_and_pages = [(_embed_to_reduced_bytes(r['embedding_vector']), r['page_id'], ) for r in batch]
    #     update_reduced_vectors_in_batch(vectors_and_pages, sqlconn)
    #     # for row in batch:
    #     #     vec = np.frombuffer(row["embedding_vector"], dtype=np.float32).reshape(
    #     #         1, -1
    #     #     )
    #     #     reduced = ipca.transform(vec).astype(np.float32).squeeze()
    #     #     update_reduced_vector_for_page(row["page_id"], reduced, sqlconn)
    #     tracker.update(1) if tracker else None

    logger.debug("Second pass: transforming and storing vectors in matrix")

    for batch in _batch_iterator(sqlconn, namespace, ["embedding_vector"], batch_size):
        # Decode the whole batch into one 2D array efficiently
        logger.debug("Creating batch matrix")
        batch_matrix = np.frombuffer(
            b"".join(row["embedding_vector"] for row in batch),
            dtype=np.float32,
        ).reshape(len(batch), -1)

        # Apply PCA transform once per batch (vectorized)
        logger.debug("Applying PCA transform to batch")
        reduced = ipca.transform(batch_matrix).astype(np.float32)

        # Convert all reduced vectors to bytes in one go
        logger.debug("Converting reduced vectors to bytes")
        reduced_bytes = [r.tobytes() for r in reduced]
        page_ids = [row["page_id"] for row in batch]

        # Combine for DB update
        logger.debug("Zipping list for database update")
        vectors_and_pages = list(zip(reduced_bytes, page_ids))
        logger.debug("Invoking update_reduced_vectors_in_batch")
        update_reduced_vectors_in_batch(namespace, vectors_and_pages, sqlconn)

        tracker.update(1) if tracker else None

    # logger.info("PCA completed and reduced vectors stored.")
    logger.debug("Done with PCA")
    return batch_counter, total_vectors


def run_umap_per_cluster(
    sqlconn: sqlite3.Connection,
    namespace: str,
    n_components: int = 3,
    limit: Optional[int] = None,
    tracker: Optional[ProgressTracker] = None,
) -> int:
    """Apply UMAP within each cluster and store 3-D vectors.

    ``limit`` caps the number of clusters processed (useful for quick tests).
    Returns the number of clusters actually processed.
    """
    # logger.info(
    #     "Running UMAP per cluster (n_components=%s, limit=%s)", n_components, limit
    # )
    # Determine distinct cluster ids that need projection
    cluster_ids_and_sizes = get_clusters_needing_projection(sqlconn, namespace, limit)

    total_vectors = sum([cluster[1] for cluster in cluster_ids_and_sizes])
    tracker.set_total(total_vectors) if tracker else None

    processed = 0
    for cluster_id, cluster_size in cluster_ids_and_sizes:
        # Gather all reduced vectors for this cluster.
        page_id_and_reduced_vectors = get_reduced_vectors_for_cluster(sqlconn, namespace, cluster_id)

        page_ids = [r[0] for r in page_id_and_reduced_vectors]
        vectors = np.vstack(
            [np.frombuffer(r[1], dtype=np.float32) for r in page_id_and_reduced_vectors]
        )
        reducer = umap.UMAP(n_components=n_components, n_neighbors=30, random_state=42,
                            min_dist=0.0, metric='cosine', init='random', verbose=False, n_jobs=1)

        three_space_vectors = reducer.fit_transform(vectors).astype(np.float32)  # type: ignore
        # Store each 3-D vector as JSON.
        for pid, vec in zip(page_ids, three_space_vectors):
            update_three_d_vector_for_page(namespace, pid, vec.tolist(), sqlconn)
        # Update centroid in cluster_info.
        # centroid = three_space_vectors.mean(axis=0).tolist()
        tracker.update(cluster_size) if tracker else None
        processed += 1
    sqlconn.commit()
    # logger.info("UMAP mapping completed for %s clusters.", processed)
    return processed


def fast_silhouette_score(X, labels, metric='cosine', sample_size=10000):
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = np.array(labels)[idx]
    else:
        X_sample = X
        labels_sample = labels
    return silhouette_score(X_sample, labels_sample, metric=metric)


def run_recursive_clustering(
    sqlconn: sqlite3.Connection,
    namespace: str,
    leaf_target: int = 50,
    max_k: int = 50,
    max_depth: int = 10,
    min_silhouette_threshold: float = 0.1,
    batch_size: int = 10_000,
    tracker: Optional[ProgressTracker] = None,
) -> int:
    """Run recursive clustering algorithm to build a tree of clusters.

    Args:
        sqlconn: SQLite database connection
        namespace: Namespace to process
        leaf_target: Target number of documents per leaf cluster
        max_k: Maximum number of clusters to create at each level
        max_depth: Maximum depth for recursion
        min_silhouette_threshold: Minimum silhouette score to continue clustering
        batch_size: Batch size for processing
        tracker: Optional progress tracker

    Returns:
        Total number of nodes created in the cluster tree
    """
    logger.debug("Starting recursive clustering with leaf_target=%s, max_k=%s, max_depth=%s",
                 leaf_target, max_k, max_depth)

    # Get total document count for the namespace
    pages_and_vectors = list(get_page_reduced_vectors(sqlconn=sqlconn, namespace=namespace))
    doc_count = len(pages_and_vectors)
    if doc_count == 0:
        logger.warning("No reduced vectors for clustering found")
        return 0

    if doc_count < max_k:
        logger.warning(f"Only found {len(pages_and_vectors)} reduced vectors, below max_k: {max_k}")
        return 0
    page_ids = [item[0] for item in pages_and_vectors]
    page_vectors = np.array([item[1] for item in pages_and_vectors])

    # Start with root node containing all documents
    root_node_id = get_cluster_tree_max_node_id(sqlconn) + 1
    logger.debug("Creating root node with ID %s", root_node_id)

    # Insert root node
    node = ClusterTreeNode(
        namespace=namespace,
        node_id=root_node_id,
        depth=0,
        doc_count=doc_count
    )
    inserted_node_id = insert_cluster_tree_node(sqlconn=sqlconn, node=node)
    logger.debug("Inserted node ID: %d", inserted_node_id)
    assert root_node_id == inserted_node_id

    # Start recursive clustering from root
    nodes_processed = _recursive_cluster_node(
        sqlconn=sqlconn,
        namespace=namespace,
        page_ids=page_ids,
        page_vectors=page_vectors,
        node_id=root_node_id,
        parent_id=None,
        depth=0,
        doc_count=doc_count,
        leaf_target=leaf_target,
        max_k=max_k,
        max_depth=max_depth,
        min_silhouette_threshold=min_silhouette_threshold,
        batch_size=batch_size,
        tracker=tracker
    )

    logger.debug("Recursive clustering completed. Total nodes processed: %s", nodes_processed)
    return nodes_processed


def _recursive_cluster_node(
    sqlconn: sqlite3.Connection,
    namespace: str,
    page_ids: list[int],
    page_vectors: NDArray,
    node_id: int,
    parent_id: Optional[int],
    depth: int,
    doc_count: int,
    leaf_target: int,
    max_k: int,
    max_depth: int,
    min_silhouette_threshold: float,
    batch_size: int,
    tracker: Optional[ProgressTracker] = None,
) -> int:
    """Recursively cluster a single node.

    Args:
        sqlconn: SQLite database connection
        namespace: Namespace to process
        node_id: Current node ID
        parent_id: Parent node ID
        depth: Current depth in the tree
        doc_count: Number of documents in this node
        leaf_target: Target number of documents per leaf cluster
        max_k: Maximum number of clusters to create at each level
        max_depth: Maximum depth for recursion
        min_silhouette_threshold: Minimum silhouette score to continue clustering
        batch_size: Batch size for processing
        tracker: Optional progress tracker

    Returns:
        Number of nodes processed (including this one)
    """
    logger.debug("Processing node %s at depth %s with %s documents", node_id, depth, doc_count)

    # Check stopping conditions
    if (doc_count <= leaf_target or
            depth >= max_depth):
        logger.debug("Node %s is a leaf (doc_count=%s, depth=%s)", node_id, doc_count, depth)
        return 1

    logger.debug("Processing %s vectors for node %s", len(page_vectors), node_id)

    # Calculate k for this node
    k = min(max_k, math.ceil(doc_count / leaf_target))
    logger.debug("Node %s: k=%s (doc_count=%s, leaf_target=%s, max_k=%s)",
                 node_id, k, doc_count, leaf_target, max_k)

    if k <= 1:
        logger.debug("Node %s cannot be split further (k=%s)", node_id, k)
        return 1

    # Perform clustering
    logger.debug("Calculating mini batch k-means for node %d", node_id)
    kmeans = MiniBatchKMeans(n_clusters=k,
                             random_state=42,
                             batch_size=batch_size,
                             max_iter=50,
                             n_init=1,
                             reassignment_ratio=0.01)
    cluster_labels = kmeans.fit_predict(page_vectors)

    # Calculate silhouette score to evaluate clustering quality
    try:
        logger.debug("Calculating silhouette score for node %d", node_id)
        silhouette_avg = fast_silhouette_score(page_vectors, cluster_labels, metric='cosine')
        logger.debug("Node %s silhouette score: %s", node_id, silhouette_avg)

        if silhouette_avg < min_silhouette_threshold:
            logger.debug("Node %s has poor clustering quality (silhouette=%s < %s), stopping recursion",
                         node_id, silhouette_avg, min_silhouette_threshold)
            return 1
    except Exception as e:
        logger.warning("Could not calculate silhouette score for node %s: %s", node_id, e)
        silhouette_avg = 0.0

    # Update node centroid
    logger.debug("Updating centroid for node %d", node_id)
    centroid = kmeans.cluster_centers_.mean(axis=0)
    centroid_blob = numpy_to_bytes(centroid)
    sqlconn.execute(
        "UPDATE cluster_tree SET centroid = ? WHERE node_id = ?",
        (centroid_blob, node_id)
    )

    # Process each child cluster
    child_nodes_processed = 0
    descendant_nodes_processed = 0

    for cluster_id in range(k):
        # Get documents belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_page_ids = [page_ids[i] for i in range(len(page_ids)) if cluster_mask[i]]
        cluster_vectors = page_vectors[cluster_mask]

        cluster_size = len(cluster_page_ids)

        logger.debug("Subcluster %d:%d has %s documents", node_id, cluster_id, cluster_size)

        if cluster_size == 0:
            continue

        # Create child node
        child_node_id = get_cluster_tree_max_node_id(sqlconn) + 1

        # Insert child node
        child_node = ClusterTreeNode(
            namespace=namespace,
            node_id=child_node_id,
            parent_id=node_id,
            depth=depth + 1,
            doc_count=cluster_size,
            sample_doc_ids=cluster_page_ids[:10]  # Sample first 10 doc IDs
        )
        inserted_node_id = insert_cluster_tree_node(sqlconn=sqlconn, node=child_node)
        child_nodes_processed += 1
        descendant_nodes_processed += 1
        logger.debug("Max node + 1: %d, inserted node id: %d", child_node_id, inserted_node_id)
        assert child_node_id == inserted_node_id

        # Recursively process child node
        recursive_descendant_nodes_processed = _recursive_cluster_node(
            sqlconn=sqlconn,
            namespace=namespace,
            page_ids=cluster_page_ids,
            page_vectors=cluster_vectors,
            node_id=child_node_id,
            parent_id=node_id,
            depth=depth + 1,
            doc_count=cluster_size,
            leaf_target=leaf_target,
            max_k=max_k,
            max_depth=max_depth,
            min_silhouette_threshold=min_silhouette_threshold,
            batch_size=batch_size,
            tracker=tracker
        )
        descendant_nodes_processed += recursive_descendant_nodes_processed

        # if child was a leaf, assign the pages to it
        if recursive_descendant_nodes_processed == 1:
            logger.debug("Processing node %d as a leaf node, adding %d pages", child_node.node_id,
                         len(cluster_page_ids))
            update_cluster_tree_assignments(sqlconn, namespace, child_node.node_id, cluster_page_ids)

    # Update child count for parent node
    update_cluster_tree_child_count(namespace, node_id, child_nodes_processed, sqlconn)

    # Update progress tracker
    if tracker:
        tracker.update(1)

    # Return total nodes processed (1 for this node + all child nodes)
    return descendant_nodes_processed
