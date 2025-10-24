from enum import StrEnum
import json
import sqlite3
import logging
from dataclasses import asdict, fields
from typing import Iterator, Optional, Type, TypeVar

import numpy as np

from numpy.typing import NDArray

from classes import Chunk, Page, PageVectors, Vector3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_tables(sqlconn: sqlite3.Connection):
    chunk_log_table_sql = """
        CREATE TABLE IF NOT EXISTS chunk_log (
            chunk_name TEXT NOT NULL PRIMARY KEY,
            namespace TEXT NOT NULL,
            chunk_archive_path TEXT,
            chunk_extracted_path TEXT,
            downloaded_at DATETIME,
            completed_at DATETIME
        );
        """
    page_log_table_sql = """
        CREATE TABLE IF NOT EXISTS page_log (
            page_id INTEGER NOT NULL PRIMARY KEY,
            title TEXT,
            chunk_name TEXT,
            url TEXT,
            extracted_at DATETIME,
            abstract TEXT
        );
        """
    # New tables for vector storage and clustering information
    page_vector_table_sql = """
        CREATE TABLE IF NOT EXISTS page_vector (
            page_id INTEGER PRIMARY KEY REFERENCES page_log(page_id) ON DELETE CASCADE,
            embedding_vector BLOB,    -- original 2048-dim numpy array (float32)
            reduced_vector BLOB,      -- PCA-reduced 100-dim array (float32)
            cluster_id INTEGER,       -- FK to cluster_info.cluster_id
            three_d_vector TEXT       -- JSON "[x, y, z]"
        );
        """
    cluster_info_table_sql = """
        CREATE TABLE IF NOT EXISTS cluster_info (
            cluster_id INTEGER NOT NULL,
            namespace TEXT NOT NULL,
            centroid_3d TEXT,         -- JSON "[x, y, z]"
            centroid_vector BLOB,     -- Full centroid vector as float32 bytes
            cluster_name TEXT,
            cluster_description TEXT,
            PRIMARY KEY (cluster_id, namespace)
        );
        """
    try:
        sqlconn.execute(chunk_log_table_sql)
        sqlconn.execute(page_log_table_sql)
        sqlconn.execute(page_vector_table_sql)
        sqlconn.execute(cluster_info_table_sql)
        # Index for fast cluster lookup
        sqlconn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_vector_cluster ON page_vector(cluster_id);"
        )
    except sqlite3.Error as e:
        logger.error(f"Failed to create tables: {e}")
        raise


class VectorType(StrEnum):
    EMBEDDING = "embedding_vector"
    REDUCED = "reduced_vector"


_sqlconns = {}


def get_sql_conn(db_file: str = "chunk_log.db") -> sqlite3.Connection:
    if _sqlconns.get(db_file):
        return _sqlconns[db_file]
    logger.info("Establishing SQLite connection")
    sqlconn = sqlite3.connect(db_file)
    sqlconn.row_factory = sqlite3.Row  # This enables dict-like access to rows
    _sqlconns[db_file] = sqlconn
    return sqlconn


T = TypeVar("T")


def _row_to_dataclass(row: sqlite3.Row, cls: Type[T]) -> T:
    """Map a Row (dict-like) to the given dataclass."""
    # Build a dict of column_name → value
    col_dict = {k: row[k] for k in row.keys()}
    # Extract only the fields that the dataclass actually defines
    field_names = {f.name for f in fields(cls)}  # type: ignore
    relevant = {k: v for k, v in col_dict.items() if k in field_names}
    return cls(**relevant)  # type: ignore[arg-type]


def get_any_page(sqlconn: sqlite3.Connection) -> Optional[Page]:
    select_a_page_sql = """
        SELECT page_id, title, chunk_name, url, extracted_at, abstract
        FROM page_log
        LIMIT 1
        """
    cursor = sqlconn.execute(select_a_page_sql)
    row = cursor.fetchone()
    return _row_to_dataclass(row, Page)


def get_page_by_id(page_id: int, sqlconn: sqlite3.Connection) -> Optional[Page]:
    select_page_sql = """
        SELECT page_id, title, chunk_name, url, extracted_at, abstract
        FROM page_log
        WHERE page_id = ?
        LIMIT 1
        """
    cursor = sqlconn.execute(select_page_sql, (page_id,))
    row = cursor.fetchone()
    if row:
        return _row_to_dataclass(row, Page)
    else:
        logger.warning(f"No page found with page_id: {page_id}")
        return None


def bytes_to_numpy(data: Optional[bytes]) -> Optional[NDArray]:
    """Convert bytes from the database back to a NumPy array."""
    return np.frombuffer(data, dtype=np.float32) if data is not None else None


def numpy_to_bytes(data: Optional[NDArray]) -> Optional[bytes]:
    """Convert NumPy array to bytes for storage."""
    return np.array(data).astype(np.float32).tobytes() if data is not None else None


def text_to_three_d_vector(data: Optional[str]) -> Optional[Vector3D]:
    """Convert JSON string to 3D vector tuple."""
    if data:
        list_data = json.loads(data)
        if not (
            isinstance(list_data, list)
            and len(list_data) == 3
            and all(isinstance(x, (int, float)) for x in list_data)
        ):
            raise ValueError(f"Invalid three_d_vector data: {data}")
        return tuple(list_data)
    else:
        return None


def three_d_vector_to_text(vector: Optional[Vector3D]) -> Optional[str]:
    """Convert 3D vector tuple to JSON string."""
    return json.dumps(vector) if vector else None


def upsert_new_page_data(page: Page, sqlconn: sqlite3.Connection) -> None:
    page_data_upsert_sql = """
        INSERT INTO page_log(page_id, title, chunk_name, url, extracted_at, abstract)
        VALUES(:page_id, :title, :chunk_name, :url, CURRENT_TIMESTAMP, :abstract)
        ON CONFLICT(page_id) DO UPDATE
        SET title = :title,
            chunk_name = :chunk_name,
            url = :url,
            extracted_at = CURRENT_TIMESTAMP,
            abstract = :abstract
        """

    try:
        cursor = sqlconn.cursor()
        cursor.execute(page_data_upsert_sql, asdict(page))
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as ex:
            logger.error(
                f"Failed to roll back sql transaction while handling another error: {ex}"
            )
        logger.error(f"Failed to upsert page data for page {page.page_id}: {e}")
        raise


def upsert_new_chunk_data(chunk: Chunk, sqlconn: sqlite3.Connection) -> None:
    upsert_sql = """
        INSERT INTO chunk_log(chunk_name, namespace, downloaded_at, completed_at)
          VALUES(:chunk_name, :namespace, NULL, NULL)
          ON CONFLICT(chunk_name) DO UPDATE
          SET chunk_name = :chunk_name, downloaded_at = NULL, completed_at = NULL;
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


def update_chunk_data(chunk: Chunk, sqlconn: sqlite3.Connection) -> None:
    update_sql = """
        UPDATE chunk_log
        SET chunk_archive_path = :chunk_archive_path,
            chunk_extracted_path = :chunk_extracted_path,
            downloaded_at = :downloaded_at,
            completed_at = :completed_at
        WHERE chunk_name = :chunk_name
        """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(update_sql, asdict(chunk))
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as ex:
            logger.error(
                f"Failed to roll back sql transaction while handling another error: {ex}"
            )
        logger.error(f"Failed to update chunk data for chunk {chunk.chunk_name}: {e}")
        raise


def get_chunk_data(chunk_name: str, sqlconn: sqlite3.Connection) -> Optional[Chunk]:
    select_sql = """
        SELECT chunk_name, namespace, chunk_archive_path, chunk_extracted_path, downloaded_at, completed_at
        FROM chunk_log
        WHERE chunk_name = :chunk_name
        LIMIT 1
        """
    cursor = sqlconn.execute(select_sql, {"chunk_name": chunk_name})
    row = cursor.fetchone()
    if row:
        return _row_to_dataclass(row, Chunk)
    else:
        logger.warning(f"No chunk found with chunk_name: {chunk_name}")
        return None


# ---------------------------------------------------------------------------
# Vector storage helper utilities (new for dimensionality processing)
# ---------------------------------------------------------------------------


def get_embedding_count(namespace: str, sqlconn: sqlite3.Connection) -> int:

    select_sql = """
        SELECT COUNT(embedding_vector)
        FROM page_vector
        INNER JOIN page_log ON page_vector.page_id = page_log.page_id
        INNER JOIN chunk_log ON chunk_log.chunk_name = page_log.chunk_name
        WHERE embedding_vector IS NOT NULL
        AND chunk_log.namespace = :namespace
    """

    cursor = sqlconn.execute(select_sql, {"namespace": namespace})
    row = cursor.fetchone()
    return row[0]


def get_reduced_vector_count(namespace: str, sqlconn: sqlite3.Connection) -> int:

    select_sql = """
        SELECT COUNT(reduced_vector)
        FROM page_vector
        INNER JOIN page_log ON page_vector.page_id = page_log.page_id
        INNER JOIN chunk_log ON chunk_log.chunk_name = page_log.chunk_name
        WHERE reduced_vector IS NOT NULL
        AND chunk_log.namespace = :namespace
    """

    cursor = sqlconn.execute(select_sql, {"namespace": namespace})
    row = cursor.fetchone()
    return row[0]


def get_page_vectors(
    page_id: int, sqlconn: sqlite3.Connection
) -> Optional[PageVectors]:
    select_sql = """
        SELECT page_id, embedding_vector, reduced_vector, cluster_id, three_d_vector
        FROM page_vector
        WHERE page_id = ?
        LIMIT 1
        """
    cursor = sqlconn.execute(select_sql, (page_id,))
    row = cursor.fetchone()
    if row:
        return _row_to_dataclass(row, PageVectors)
    else:
        logger.warning(f"No page vectors found with page_id: {page_id}")
        return None


def store_vector(
    page_id: int, column: VectorType, np_array: NDArray, sqlconn: sqlite3.Connection
) -> None:
    """
    Serialise a NumPy array as float32 bytes and store it in *column*.
    """

    blob = np_array.astype(np.float32).tobytes()
    sql = f"UPDATE page_vector SET {column} = :blob WHERE page_id = :page_id"
    logger.debug(f"SQL update statement: {sql}")
    try:
        sqlconn.execute(sql, {"blob": blob, "page_id": page_id})
        sqlconn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to store blob for page {page_id}: {e}")
        raise


def store_three_d_vector(
    page_id: int, vector: Vector3D, sqlconn: sqlite3.Connection
) -> None:
    """Serialise *data* to JSON and store it in *column*.

    Currently used for ``three_d_vector`` (JSON array ``[x, y, z]``).
    """
    json_str = json.dumps(vector)
    sql = "UPDATE page_vector SET three_d_vector = ? WHERE page_id = ?"
    try:
        sqlconn.execute(sql, (json_str, page_id))
        sqlconn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to store JSON for page {page_id}: {e}")
        raise


def get_page_embeddings(
    sqlconn: sqlite3.Connection,
) -> Iterator[tuple[int, Optional[NDArray]]]:
    """
    Yield tuples containing ``page_id`` and ``embedding_vector`` in NumPy array form.
    """
    sql = "SELECT page_id, embedding_vector FROM page_vector"
    cursor = sqlconn.execute(sql)
    for row in cursor:
        page_id = row["page_id"]
        vector_blob = row["embedding_vector"]
        vector = bytes_to_numpy(vector_blob)
        yield (page_id, vector)


def get_page_reduced_vectors(
    sqlconn: sqlite3.Connection,
) -> Iterator[tuple[int, Optional[NDArray]]]:
    """
    Yield tuples containing ``page_id`` and ``reduced_vector`` in NumPy array form.
    """
    sql = "SELECT page_id, reduced_vector FROM page_vector"
    cursor = sqlconn.execute(sql)
    for row in cursor:
        page_id = row["page_id"]
        vector_blob = row["reduced_vector"]
        vector = bytes_to_numpy(vector_blob)
        yield (page_id, vector)


def update_embeddings_for_page(
    page_id: int, embedding_vector: NDArray, sqlconn: sqlite3.Connection
) -> None:
    # convert numpy arrays to bytes for storage
    embedding_vector_bytes = numpy_to_bytes(embedding_vector)
    prepared_data = {"page_id": page_id, "embedding_vector": embedding_vector_bytes}
    update_page_vector_sql = """
        INSERT INTO page_vector (page_id, embedding_vector) VALUES (:page_id, :embedding_vector)
        ON CONFLICT(page_id) DO
        UPDATE SET embedding_vector = :embedding_vector WHERE page_id = :page_id;
        """

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
        logger.exception(f"Failed to update embedding vector for page {page_id}: {e}")
        raise


def update_reduced_vector_for_page(
    page_id: int, reduced_vector: NDArray, sqlconn: sqlite3.Connection
) -> None:
    # convert numpy arrays to bytes for storage
    reduced_vector_bytes = numpy_to_bytes(reduced_vector)
    prepared_data = {"page_id": page_id, "reduced_vector": reduced_vector_bytes}
    update_page_vector_sql = "UPDATE page_vector SET reduced_vector = :reduced_vector WHERE page_id = :page_id;"

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
        logger.exception(f"Failed to update reduced vector for page {page_id}: {e}")
        raise


def update_three_d_vector_for_page(
    page_id: int, vector: Vector3D, sqlconn: sqlite3.Connection
) -> None:
    # convert vector to text for storage
    vector_text = three_d_vector_to_text(vector)
    prepared_data = {"page_id": page_id, "three_d_vector": vector_text}
    update_page_vector_sql = (
        "UPDATE page_log SET three_d_vector = :vector_text WHERE page_id = :page_id;"
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
        logger.exception(f"Failed to update reduced vector for page {page_id}: {e}")
        raise


def update_cluster_centroid(
    cluster_id: int,
    namespace: str,
    centroid_vector: NDArray,
    sqlconn: sqlite3.Connection,
) -> None:
    centroid_blob = numpy_to_bytes(centroid_vector)
    try:
        sqlconn.execute(
            "UPDATE cluster_info SET centroid_vector = ? WHERE cluster_id = ? AND namespace = ?",
            (centroid_blob, int(cluster_id), namespace),
        )
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e1:
            logger.error(
                f"Failed to roll back sql transaction while handling another error: {e1}"
            )
            pass
        logger.exception(
            f"Failed to update cluster centroid for cluster {cluster_id} in namespace {namespace}: {e}"
        )
        raise


def get_page_ids_needing_embedding_for_chunk(
    chunk_name: str, sqlconn: sqlite3.Connection
) -> list[int]:
    select_sql = """
        SELECT page_id
        FROM page_log
        LEFT OUTER JOIN page_vector USING(page_id)
        WHERE chunk_name = :chunk_name
        AND embedding_vector IS NULL
        ORDER BY page_id ASC;
    """
    cursor = sqlconn.execute(select_sql, {"chunk_name": chunk_name})
    rows = cursor.fetchall()
    page_id_list = [row["page_id"] for row in rows]
    return page_id_list
