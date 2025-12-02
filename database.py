import json
import os
import sqlite3
import logging
from dataclasses import asdict, fields
from typing import Iterable, Iterator, Optional, Type, TypeVar

import numpy as np

from numpy.typing import NDArray

from classes import Chunk, ClusterTreeNode, Page, PageContent, ClusterNodeTopics

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ensure_tables(sqlconn: sqlite3.Connection):
    chunk_log_table_sql = """
        CREATE TABLE IF NOT EXISTS chunk_log (
            namespace TEXT NOT NULL,
            chunk_name TEXT NOT NULL,
            chunk_archive_path TEXT,
            chunk_extracted_path TEXT,
            downloaded_at DATETIME,
            unpacked_at DATETIME,
            PRIMARY KEY (namespace, chunk_name)
        );
        """
    """
    INSERT INTO chunk_log_2 (namespace, chunk_name, chunk_archive_path,
            chunk_extracted_path, downloaded_at, unpacked_at)
    SELECT namespace, chunk_name, chunk_archive_path, chunk_extracted_path, downloaded_at, unpacked_at
    FROM chunk_log;
    """
    page_log_table_sql = """
        CREATE TABLE IF NOT EXISTS page_log (
            namespace TEXT NOT NULL,
            page_id INTEGER NOT NULL,
            title TEXT,
            chunk_name TEXT,
            url TEXT,
            extracted_at DATETIME,
            abstract TEXT,
            PRIMARY KEY (namespace, page_id)
        );
        """
    """
    INSERT INTO page_log_2 (namespace, page_id, title, chunk_name, url, extracted_at, abstract)
    SELECT 'enwiki_namespace_0', page_id, title, chunk_name, url, extracted_at, abstract
    FROM page_log;
    """
    # New tables for vector storage and clustering information
    page_vector_table_sql = """
        CREATE TABLE IF NOT EXISTS page_vector (
            namespace TEXT NOT NULL,
            page_id INTEGER NOT NULL REFERENCES page_log(page_id) ON DELETE CASCADE,
            embedding_vector BLOB,    -- original 2048-dim numpy array (float32)
            reduced_vector BLOB,      -- PCA-reduced 100-dim array (float32)
            cluster_id INTEGER,       -- FK to cluster_info.cluster_id
            cluster_node_id INTEGER,  -- FK to cluster_tree.node_id
            three_d_vector TEXT,       -- JSON "[x, y, z]"
            PRIMARY KEY (namespace, page_id)
        );
        """
    """
    INSERT INTO page_vector_2 (namespace, page_id, embedding_vector, reduced_vector,
             cluster_id, cluster_node_id, three_d_vector)
    SELECT 'enwiki_namespace_0', page_id, embedding_vector, reduced_vector, cluster_id, NULL, three_d_vector
    FROM page_vector;
    """
    cluster_tree_table_sql = """
        CREATE TABLE IF NOT EXISTS cluster_tree (
            namespace TEXT,
            node_id BIGINT ,
            parent_id BIGINT,
            depth SMALLINT NOT NULL,
            centroid BLOB,
            doc_count INT NOT NULL,
            top_terms TEXT,
            sample_doc_ids TEXT,
            child_count INT DEFAULT 0,
            first_label TEXT,
            final_label TEXT,
            PRIMARY KEY (namespace, node_id),
            FOREIGN KEY (namespace, parent_id) REFERENCES cluster_tree(namespace, node_id)
        );
        """
    try:
        sqlconn.execute(chunk_log_table_sql)
        sqlconn.execute(page_log_table_sql)
        sqlconn.execute(page_vector_table_sql)
        sqlconn.execute(cluster_tree_table_sql)
        # Index for fast cluster lookup
        sqlconn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_vector_ns_cluster ON page_vector(namespace, cluster_id);"
        )
        # Indexes for cluster_tree table
        sqlconn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cluster_tree_ns_parent ON cluster_tree(namespace, parent_id, depth);"
        )
        sqlconn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cluster_tree_ns_depth ON cluster_tree(namespace, depth);"
        )

        # performance pragmas

    except sqlite3.Error as e:
        logger.error(f"Failed to create tables: {e}")
        raise


_sqlconns = {}


def _get_sql_conn_for_file(db_file: str = "chunk_log.db") -> sqlite3.Connection:
    # if we already created a connection, just return that
    if _sqlconns.get(db_file):
        return _sqlconns[db_file]

    # otherwise, make a new connection
    logger.info("Establishing SQLite connection to %s", db_file)
    sqlconn = sqlite3.connect(db_file)
    sqlconn.row_factory = sqlite3.Row  # This enables dict-like access to rows

    # Performance pragmas
    try:
        sqlconn.execute("PRAGMA journal_mode=WAL;")
        sqlconn.execute("PRAGMA synchronous=NORMAL;")
        sqlconn.execute("PRAGMA temp_store = MEMORY;")
        sqlconn.execute("PRAGMA cache_size = -20000;")  # ~20MB cache (adjust as needed)
    except sqlite3.Error:
        pass

    # cache the connection for reuse later
    _sqlconns[db_file] = sqlconn
    return sqlconn


def get_sql_conn(namespace: str, db_directory: str = "") -> sqlite3.Connection:
    # convert the namespace to a db file path
    db_file_path = os.path.join(db_directory, f"{namespace}.db")
    logger.debug("Namespace: %s, db file: %s", namespace, db_file_path)
    return _get_sql_conn_for_file(db_file_path)


T = TypeVar("T")


def _row_to_dataclass(row: sqlite3.Row, cls: Type[T]) -> T:
    """Map a Row (dict-like) to the given dataclass."""
    # Build a dict of column_name → value
    col_dict = {k: row[k] for k in row.keys()}
    # Extract only the fields that the dataclass actually defines
    field_names = {f.name for f in fields(cls)}  # type: ignore
    relevant = {k: v for k, v in col_dict.items() if k in field_names}
    return cls(**relevant)  # type: ignore[arg-type]


def get_page_by_id(namespace: str, page_id: int, sqlconn: sqlite3.Connection) -> Optional[Page]:
    select_page_sql = """
        SELECT namespace, page_id, title, chunk_name, url, extracted_at, abstract
        FROM page_log
        WHERE namespace = ? AND page_id = ?
        LIMIT 1
        """
    cursor = sqlconn.execute(select_page_sql, (namespace, page_id,))
    row = cursor.fetchone()
    if row:
        return _row_to_dataclass(row, Page)
    else:
        logger.warning("No page found in namespace %s with page_id %d", namespace, page_id)
        return None


def bytes_to_numpy(data: Optional[bytes]) -> Optional[NDArray]:
    """Convert bytes from the database back to a NumPy array."""
    return np.frombuffer(data, dtype=np.float32) if data is not None else None


def numpy_to_bytes(data: Optional[NDArray]) -> Optional[bytes]:
    """Convert NumPy array to bytes for storage."""
    return np.array(data).astype(np.float32).tobytes() if data is not None else None


def text_to_three_d_vector(data: Optional[str]) -> Optional[NDArray]:
    """Convert JSON string to 3D vector tuple."""
    if data:
        list_data = json.loads(data)
        if not (
            isinstance(list_data, list)
            and len(list_data) == 3
            and all(isinstance(x, (int, float)) for x in list_data)
        ):
            raise ValueError(f"Invalid three_d_vector data: {data}")
        return np.fromiter(list_data, dtype=np.float32)
    else:
        return None


def three_d_vector_to_text(vector: Optional[NDArray]) -> Optional[str]:
    """Convert 3D vector tuple to JSON string."""
    if vector is None:
        return None
    if isinstance(vector, list):
        # Handle list input
        try:
            # Check for NaN or infinite values
            processed_vector = []
            for x in vector:
                if isinstance(x, (int, float)):
                    if np.isfinite(x):
                        processed_vector.append(float(x))
                    else:
                        # Replace NaN/infinite with 0
                        processed_vector.append(0.0)
                else:
                    processed_vector.append(0.0)
            return json.dumps(processed_vector)
        except (ValueError, TypeError):
            return json.dumps([0.0, 0.0, 0.0])  # Fallback to zero vector
    else:
        # Handle numpy array input
        try:
            # Check for NaN or infinite values
            if not np.all(np.isfinite(vector)):
                logger.warning(f"Vector contains NaN or infinite values: {vector}")
                # Replace NaN/infinite with 0
                vector = np.where(np.isfinite(vector), vector, 0.0)
            return json.dumps(vector.astype(np.float32).tolist())
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert vector to text: {vector}")
            return json.dumps([0.0, 0.0, 0.0])  # Fallback to zero vector


def upsert_new_pages_in_batch(pages: list[Page], sqlconn: sqlite3.Connection, batch_size: int = 1000):
    sql = """
      INSERT INTO page_log(namespace, page_id, title, chunk_name, url, extracted_at, abstract)
      VALUES(:namespace, :page_id, :title, :chunk_name, :url, CURRENT_TIMESTAMP, :abstract)
      ON CONFLICT(namespace, page_id) DO UPDATE SET
        title = :title,
        chunk_name = :chunk_name,
        url = :url,
        extracted_at = CURRENT_TIMESTAMP,
        abstract = :abstract
    """
    cursor = sqlconn.cursor()
    try:
        for i in range(0, len(pages), batch_size):
            batch = pages[i: i + batch_size]
            params = [asdict(p) for p in batch]
            cursor.executemany(sql, params)
            sqlconn.commit()
    except sqlite3.Error:
        sqlconn.rollback()
        raise


def upsert_new_chunk_data(chunk: Chunk, sqlconn: sqlite3.Connection) -> None:
    upsert_sql = """
        INSERT INTO chunk_log(chunk_name, namespace, downloaded_at, unpacked_at)
          VALUES(:chunk_name, :namespace, NULL, NULL)
          ON CONFLICT(chunk_name, namespace) DO UPDATE
          SET downloaded_at = NULL, unpacked_at = NULL;
        """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(
            upsert_sql, {"chunk_name": chunk.chunk_name, "namespace": chunk.namespace}
        )
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as ex:
            logger.error(
                f"Failed to roll back sql transaction while handling another error: {ex}"
            )
        logger.error(f"Failed to upsert chunk info for chunk {chunk.chunk_name}: {e}")
        raise


# ---------------------------------------------------------------------------
# Vector storage helper utilities (new for dimensionality processing)
# ---------------------------------------------------------------------------


def get_embedding_count(namespace: str, sqlconn: sqlite3.Connection) -> int:

    select_sql = """
        SELECT COUNT(embedding_vector)
        FROM page_vector
        WHERE embedding_vector IS NOT NULL
        AND namespace = :namespace
    """

    cursor = sqlconn.execute(select_sql, {"namespace": namespace})
    row = cursor.fetchone()
    return row[0]


def get_page_reduced_vectors(
    sqlconn: sqlite3.Connection,
    namespace: str
) -> Iterator[tuple[int, Optional[NDArray]]]:
    """
    Yield tuples containing ``page_id`` and ``reduced_vector`` in NumPy array form.
    """
    sql = """
        SELECT page_id, reduced_vector
        FROM page_vector
        WHERE namespace = ?
        AND reduced_vector IS NOT NULL
        """
    cursor = sqlconn.execute(sql, (namespace,))
    for row in cursor:
        page_id = row["page_id"]
        vector_blob = row["reduced_vector"]
        vector = bytes_to_numpy(vector_blob)
        yield (page_id, vector)


def upsert_embeddings_in_batch(
    namespace: str,
    pages: list[tuple[int, NDArray]],
    sqlconn: sqlite3.Connection,
    batch_size: int = 1000
):
    page_dict_list = [
        {
            "namespace": namespace,
            "page_id": page[0],
            "embedding_vector": numpy_to_bytes(page[1])
         }
        for page in pages
    ]

    update_page_vector_sql = """
        INSERT INTO page_vector (namespace, page_id, embedding_vector)
        VALUES (:namespace, :page_id, :embedding_vector)
        ON CONFLICT(namespace, page_id) DO
        UPDATE SET embedding_vector = :embedding_vector WHERE page_id = :page_id;
        """

    cursor = sqlconn.cursor()
    try:
        for i in range(0, len(page_dict_list), batch_size):
            batch = page_dict_list[i: i + batch_size]
            params = batch
            cursor.executemany(update_page_vector_sql, params)
            sqlconn.commit()
    except sqlite3.Error:
        sqlconn.rollback()
        raise


def update_reduced_vectors_in_batch(
        namespace: str,
        reduced_vector_bytes_and_page_id_list: Iterable[tuple[bytes, int]],
        sqlconn: sqlite3.Connection
) -> None:
    update_page_vector_sql = f"""
        UPDATE page_vector
        SET reduced_vector = ?
        WHERE namespace = '{namespace}' AND page_id = ?;
    """

    try:
        logger.debug("Updating reduced vectors in batch")
        cursor = sqlconn.cursor()
        cursor.executemany(update_page_vector_sql, reduced_vector_bytes_and_page_id_list)
        logger.debug("Committing reduced vectors in batch")
        sqlconn.commit()
        logger.debug("Completed update of reduced in batch")
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(
                f"Failed to roll back sql transaction while handling another error: {e1}"
            )
            pass
        logger.exception(f"Failed to update reduced vectors in batch: {e}")
        raise


def update_three_d_vector_for_page(
    namespace: str,
    page_id: int,
    vector: NDArray,
    sqlconn: sqlite3.Connection
) -> None:
    # convert vector to text for storage
    # Handle both numpy arrays and lists
    if isinstance(vector, list):
        # Convert list to numpy array for consistent processing
        vector_array = np.array(vector, dtype=np.float32)
    else:
        # Already a numpy array
        vector_array = vector

    vector_text = three_d_vector_to_text(vector_array)

    # Debug logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Page %d: vector type=%s, vector_text type=%s, vector_text=%s",
            page_id, type(vector), type(vector_text), repr(vector_text)
        )

    if vector_text is None:
        logger.warning(f"Vector text is None for page {page_id}, vector={vector}")
        # vector_text = json.dumps([0.0, 0.0, 0.0])  # Fallback to zero vector

    prepared_data = {"namespace": namespace, "page_id": page_id, "vector_text": vector_text}
    update_page_vector_sql = (
        """
        UPDATE page_vector
        SET three_d_vector = :vector_text
        WHERE namespace = :namespace AND page_id = :page_id;
        """
    )

    try:
        cursor = sqlconn.cursor()
        cursor.execute(update_page_vector_sql, prepared_data)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(
                f"Failed to roll back sql transaction while handling another error: {e1}"
            )
            pass
        logger.exception(f"Failed to update three_d_vector for page {page_id}: {e}")
        raise


def get_page_ids_needing_embedding_for_chunk(
    chunk_name: str,
    sqlconn: sqlite3.Connection,
    namespace: Optional[str]
) -> list[int]:
    select_sql = """
        SELECT page_id
        FROM page_log
        LEFT OUTER JOIN page_vector USING(namespace, page_id)
        WHERE chunk_name = :chunk_name
        AND embedding_vector IS NULL
        ORDER BY page_id ASC;
    """
    cursor = sqlconn.execute(select_sql, {"chunk_name": chunk_name})
    rows = cursor.fetchall()
    page_id_list = [row["page_id"] for row in rows]
    return page_id_list


def get_clusters_needing_projection(
    sqlconn: sqlite3.Connection,
    namespace: str,
    limit: Optional[int],
) -> list[tuple[int, int]]:
    select_sql = f"""
        SELECT cluster_id, COUNT(page_id)
        FROM page_vector
        WHERE three_d_vector IS NULL
        AND cluster_id IS NOT NULL
        AND namespace = :namespace
        GROUP BY cluster_id
        ORDER BY cluster_id ASC
        {f'LIMIT {limit}' if limit else ''}
        """
    cursor = sqlconn.execute(select_sql, {'namespace': namespace})
    rows = cursor.fetchall()
    return [(row[0], row[1]) for row in rows]


def get_reduced_vectors_for_cluster(
    sqlconn: sqlite3.Connection,
    namespace: str,
    cluster_id: int
) -> list[tuple[int, bytes]]:
    select_sql = """
        SELECT page_id, reduced_vector
        FROM page_vector
        WHERE cluster_id = :cluster_id
        AND namespace = :namespace
        ORDER BY page_id ASC
        """
    cursor = sqlconn.execute(select_sql, {'namespace': namespace, 'cluster_id': cluster_id})
    rows = cursor.fetchall()
    return rows


# ---------------------------------------------------------------------------
# Helper functions for cluster_tree operations
# ---------------------------------------------------------------------------

def insert_cluster_tree_node(
    sqlconn: sqlite3.Connection,
    node: ClusterTreeNode
) -> int:
    """Insert a new node into the cluster_tree table."""
    centroid_blob = numpy_to_bytes(node.centroid) if node.centroid is not None else None
    top_terms_json = json.dumps(node.top_terms) if node.top_terms else None
    sample_doc_ids_json = json.dumps(node.sample_doc_ids) if node.sample_doc_ids else None

    sql = """
        INSERT INTO cluster_tree (
            node_id, namespace, parent_id, child_count, depth, centroid, doc_count, top_terms, sample_doc_ids,
            first_label, final_label
        )
        VALUES (
            :node_id, :namespace, :parent_id, :child_count, :depth, :centroid, :doc_count, :top_terms, :sample_doc_ids,
            :first_label, :final_label
        )
        """
    try:
        node_dict = asdict(node)
        node_dict['centroid'] = centroid_blob
        node_dict['top_terms'] = top_terms_json
        node_dict['sample_doc_ids'] = sample_doc_ids_json
        cursor = sqlconn.cursor()
        cursor.execute(sql, node_dict)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to insert cluster_tree node {node.node_id} in namespace {node.namespace}: {e}")
        raise

    try:
        cursor = sqlconn.execute("SELECT last_insert_rowid()")
        row = cursor.fetchone()
        return row[0]
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve most recently generated cluster_tree ID: {e}")
        raise


def update_cluster_tree_child_count(
    namespace: str,
    node_id: int,
    child_count: int,
    sqlconn: sqlite3.Connection
) -> None:
    """Update the child_count for a cluster_tree node."""
    sql = "UPDATE cluster_tree SET child_count = ? WHERE namespace = ? AND node_id = ?"
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (child_count, namespace, node_id))
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to update child_count for cluster_tree node {node_id}: {e}")
        raise


def get_cluster_tree_max_node_id(sqlconn: sqlite3.Connection) -> int:
    """Get the maximum node_id from cluster_tree."""
    sql = "SELECT MAX(node_id) FROM cluster_tree"
    cursor = sqlconn.execute(sql)
    row = cursor.fetchone()
    return row[0] if row[0] else 0


def update_cluster_tree_first_labels(
        sqlconn: sqlite3.Connection,
        namespace: str,
        cluster_nodes: Iterable[ClusterNodeTopics]
        ) -> None:
    """Update the first_label for the given cluster tree node."""
    sql = """
        UPDATE cluster_tree
        SET first_label = ?
        WHERE namespace = ? AND node_id = ?
    """
    try:
        params = [(n.first_label, namespace, n.node_id, ) for n in cluster_nodes]
        cursor = sqlconn.cursor()
        cursor.executemany(sql, params)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to update first labels for namespace {namespace}: {e}")
        raise


def update_cluster_tree_final_labels(
        sqlconn: sqlite3.Connection,
        namespace: str,
        cluster_nodes: Iterable[ClusterNodeTopics]
        ) -> None:
    """Update the final_label for the given cluster tree node."""
    sql = """
        UPDATE cluster_tree
        SET final_label = ?
        WHERE namespace = ? AND node_id = ?
    """
    try:
        params = [(n.final_label, namespace, n.node_id) for n in cluster_nodes]
        cursor = sqlconn.cursor()
        cursor.executemany(sql, params)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to update final labels for namespace {namespace}: {e}")
        raise


def update_cluster_tree_assignments(
        sqlconn: sqlite3.Connection,
        namespace: str,
        cluster_node_id: int,
        cluster_page_ids: list[int],
        batch_size: int = 1000
        ):
    """Assign a list of page IDs to a cluster tree node"""

    sql = """
        UPDATE page_vector
        SET cluster_node_id = ?
        WHERE namespace = ? AND page_id = ?;
    """
    try:
        cursor = sqlconn.cursor()
        for i in range(0, len(cluster_page_ids), batch_size):
            batch = cluster_page_ids[i: i + batch_size]
            params = [(cluster_node_id, namespace, page_id) for page_id in batch]
            cursor.executemany(sql, params)
            sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to update cluster_node_id assignments for cluster node {cluster_node_id}: {e}")
        raise


def delete_cluster_tree(sqlconn: sqlite3.Connection, namespace: str):
    """Delete the existing cluster tree nodes and the cluster node assignments for pages for a given namespace."""
    page_vector_delete_sql = """
        UPDATE page_vector
        SET cluster_node_id = NULL
        WHERE namespace = ?;
    """
    cluster_node_delete_sql = """
        DELETE FROM cluster_tree
        WHERE namespace = ?;
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(page_vector_delete_sql, (namespace,))
        cursor.execute(cluster_node_delete_sql, (namespace,))
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to delete cluster tree for namespace {namespace}: {e}")
        raise


def get_pages_in_cluster(sqlconn: sqlite3.Connection, namespace: str, node_id: int) -> list[Page]:
    """Get the list of page_ids for the given cluster node"""
    sql = """
        SELECT pl.namespace, pl.page_id, pl.title, pl.chunk_name, pl.url, pl.extracted_at, pl.abstract
        FROM page_log pl
        INNER JOIN page_vector pv on pl.namespace = pv.namespace and pl.page_id = pv.page_id
        WHERE pl.namespace = ?
        AND pv.cluster_node_id = ?
        ORDER BY pl.page_id ASC
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, node_id, ))
        rows = cursor.fetchall()
        return [_row_to_dataclass(row, Page) for row in rows]

    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to get pages for cluster node {node_id} in namespace {namespace}: {e}")
        raise


def get_pages_in_all_clusters(sqlconn: sqlite3.Connection,
                              namespace: str
                              ) -> dict[int, list[PageContent]]:
    """Get the map of node_id to tuple of (page_id, title, abstract) for the given namespace"""
    # Trim abstracts to 500 characters
    sql = """
        SELECT pv.cluster_node_id, pl.page_id, pl.title, SUBSTR(pl.abstract, 1, 500)
        FROM page_log pl
        INNER JOIN page_vector pv on pl.namespace = pv.namespace and pl.page_id = pv.page_id
        WHERE pl.namespace = ?
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, ))
        results = dict()
        while True:
            rows = cursor.fetchmany(100)
            if not rows:
                break
            for row in rows:
                node_id = row[0]
                page_content = PageContent(
                    page_id=row[1],
                    title=row[2],
                    abstract=row[3]
                )
                if node_id not in results:
                    results[node_id] = list()
                results[node_id].append(page_content)
        return results

    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to get pages for all clusters in namespace {namespace}: {e}")
        raise


def get_cluster_parent_id(sqlconn: sqlite3.Connection, namespace: str, node_id: int) -> Optional[int]:
    """Get the id of the cluster node's parent node."""
    sql = """
        SELECT parent_id
        FROM cluster_tree
        WHERE namespace = ?
        AND node_id = ?
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, node_id,))
        row = cursor.fetchone()
        return row[0]

    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to get parent id for cluster in {namespace} with node id {node_id}: {e}")
        raise


def get_neighboring_first_topics(sqlconn: sqlite3.Connection,
                                 namespace: str,
                                 node_id: int,
                                 parent_node_id: int) -> list[str]:
    """Get the list of first_topics for the given cluster node"""
    sql = """
        SELECT first_label
        FROM cluster_tree
        WHERE namespace = ?
        AND parent_id = ?
        AND node_id <> ?
        AND first_label IS NOT NULL
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, parent_node_id, node_id))
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to get list of first_topics for cluster node {node_id} in namespace {namespace}: {e}")
        raise


def get_cluster_node_first_pass_topic(sqlconn: sqlite3.Connection, namespace: str, node_id: int) -> str:
    """Get the first pass topic for a given cluster node"""
    sql = """
        SELECT first_label
        FROM cluster_tree
        WHERE namespace = ?
        AND node_id = ?
        ORDER BY node_id ASC
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, node_id, ))
        row = cursor.fetchone()
        return row[0]

    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to get first topic for cluster node {node_id} in namespace {namespace}: {e}")
        raise


def get_cluster_final_topics(sqlconn: sqlite3.Connection, namespace: str) -> list[tuple[int, int, str, int]]:
    sql = """
        SELECT parent_id, node_id, first_label, depth
        FROM cluster_tree
        WHERE namespace = ?
        ORDER BY parent_id ASC
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, ))
        rows = cursor.fetchall()
        return [(row[0], row[1], row[2], row[3], ) for row in rows]

    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to get final topics for cluster in namespace {namespace}: {e}")
        raise


def remove_existing_cluster_topics(sqlconn: sqlite3.Connection, namespace: str) -> None:
    """Remove existing topics for cluster """
    sql = """
        UPDATE cluster_tree
        SET first_label = NULL,
            final_label = NULL
        WHERE namespace = ?
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace,))
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.exception(f"Failed to remove labels on cluster tree for namespace {namespace}: {e}")
        raise


def get_all_cluster_nodes_for_topic_labeling(sqlconn: sqlite3.Connection,
                                             namespace: str) -> list[ClusterNodeTopics]:
    """
    Load all the cluster tree nodes at once for topic labeling.
    Marks no nodes as leaf nodes yet, that must be done after.

    :param: sqlconn   The Sqlite3 connection to use
    :param: namespace  The namespace for the cluster tree
    :return: A list of cluster node topic objects
    """
    sql = """
        SELECT node_id, depth, parent_id, first_label, final_label
        FROM cluster_tree
        WHERE namespace = ?;
    """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(sql, (namespace, ))
        result = [ClusterNodeTopics(
            node_id=row[0],
            depth=row[1],
            parent_id=row[2],
            first_label=row[3],
            final_label=row[4],
            is_leaf=False  # mark none as leaf now, find leaves later
        ) for row in cursor.fetchall()]
        return result
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e1}")
            pass
        logger.error(f"Failed to get cluster nodes for namespace `{namespace}`: {e}")
        raise
