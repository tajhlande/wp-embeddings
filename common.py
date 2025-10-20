import json
import sqlite3
import logging

from numpy.typing import NDArray

import numpy as np

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
            abstract TEXT,
            embedding_vector BLOB,
            reduced_vector BLOB,
            three_d_vector TEXT
        );
        """
    try:
        sqlconn.execute(chunk_log_table_sql)
        sqlconn.execute(page_log_table_sql)
    except sqlite3.Error as e:
        logger.error(f"Failed to create tables: {e}")
        raise

_sqlconns = {}

def get_sql_conn(db_file: str = "chunk_log.db") -> sqlite3.Connection:
    if _sqlconns.get(db_file):
        return _sqlconns[db_file]
    logger.info("Establishing SQLite connection")
    sqlconn = sqlite3.connect(db_file)
    sqlconn.row_factory = sqlite3.Row  # This enables dict-like access to rows
    _sqlconns[db_file] = sqlconn
    return sqlconn

def _get_page_from_row(row: sqlite3.Row) -> dict:
    return {
    'page_id': row['page_id'],
        'title': row['title'],         
        'chunk_name': row['chunk_name'],
        'url': row['url'],
        'extracted_at': row['extracted_at'],
        'abstract': row['abstract']  ,
        'embedding_vector': row['embedding_vector'],
        'reduced_vector': row['reduced_vector'],
        'three_d_vector': row['three_d_vector']
    }

def get_any_page(sqlconn: sqlite3.Connection) -> dict:
    select_a_page_sql = """
        SELECT page_id, title, chunk_name, url, extracted_at, abstract, embedding_vector, reduced_vector, three_d_vector
        FROM page_log
        LIMIT 1
        """
    cursor = sqlconn.execute(select_a_page_sql) 
    row = cursor.fetchone()
    page_data = _get_page_from_row(row)
    return page_data

def get_page_by_id(page_id: int, sqlconn: sqlite3.Connection) -> dict:
    select_page_sql = """
        SELECT page_id, title, chunk_name, url, extracted_at, abstract, embedding_vector, reduced_vector, three_d_vector
        FROM page_log
        WHERE page_id = ?
        LIMIT 1
        """
    cursor = sqlconn.execute(select_page_sql, (page_id,)) 
    row = cursor.fetchone()
    if row:
        fetched_data = _get_page_from_row(row)
    else:
        logger.warning(f"No page found with page_id: {page_id}")
        return {}
    
    # convert stored bytes to numpy arrays 
    embedding_vector = np.frombuffer(fetched_data['embedding_vector'], dtype=np.float32) if fetched_data.get('embedding_vector') else None
    reduced_vector = np.frombuffer(fetched_data['reduced_vector'], dtype=np.float32) if fetched_data.get('reduced_vector') else None
    three_d_vector = json.loads(fetched_data['three_d_vector']) if fetched_data.get('three_d_vector') else None
    page_data = {
        key: embedding_vector if key == 'embedding_vector' else 
             reduced_vector if key == 'reduced_vector' else
             three_d_vector if key == 'three_d_vector' else
             value
        for key, value in fetched_data.items()
    }
    return page_data
    
def upsert_new_page_data(page_data: dict, sqlconn: sqlite3.Connection) -> None:
    page_data_upsert_sql = """
        INSERT INTO page_log(page_id, title, chunk_name, url, extracted_at, abstract, embedding_vector, reduced_vector, three_d_vector) 
        VALUES(:page_id, :title, :chunk_name, :url, CURRENT_TIMESTAMP, :abstract, NULL, NULL, NULL)
        ON CONFLICT(page_id) DO UPDATE 
        SET title = :title,
            chunk_name = :chunk_name,
            url = :url,
            extracted_at = CURRENT_TIMESTAMP,           
            abstract = :abstract, 
            embedding_vector = NULL,
            reduced_vector = NULL,
            three_d_vector = NULL
        """

    try:
        cursor = sqlconn.cursor()
        cursor.execute(page_data_upsert_sql, page_data)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as ex:
            logger.error(f"Failed to roll back sql transaction while handling another error: {ex}")            
        logger.error(f"Failed to upsert page data for page {page_data.get('page_id')}: {e}")
        raise

def upsert_new_chunk_data(chunk_name: str, namespace: str, sqlconn: sqlite3.Connection) -> dict:
    upsert_sql = """
        INSERT INTO chunk_log(chunk_name, namespace, downloaded_at, completed_at)
          VALUES(:chunk_name, :namespace, NULL, NULL)
          ON CONFLICT(chunk_name) DO UPDATE 
          SET chunk_name = :chunk_name, downloaded_at = NULL, completed_at = NULL;
        """
    try:
        cursor = sqlconn.cursor()
        cursor.execute(upsert_sql, { 'chunk_name': chunk_name, 'namespace': namespace})
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as ex:
            logger.error(f"Failed to roll back sql transaction while handling another error: {ex}")            
        logger.error(f"Failed to upsert chunk info for chunk {chunk_name}: {e}")
        raise
    return {
        'chunk_name': chunk_name,
        'namespace': namespace,
        'downloaded_at': None,
        'completed_at': None,
        'chunk_archive_path': None,
        'chunk_extracted_path': None
    }

def update_chunk_data(chunk_data: dict, sqlconn: sqlite3.Connection) -> None:
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
        cursor.execute(update_sql, chunk_data)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as ex:
            logger.error(f"Failed to roll back sql transaction while handling another error: {ex}")            
        logger.error(f"Failed to update chunk data for chunk {chunk_data.get('chunk_name')}: {e}")
        raise

def get_chunk_data(chunk_name: str, sqlconn: sqlite3.Connection) -> dict:
    select_sql = """
        SELECT chunk_name, namespace, chunk_archive_path, chunk_extracted_path, downloaded_at, completed_at
        FROM chunk_log
        WHERE chunk_name = :chunk_name
        LIMIT 1
        """
    cursor = sqlconn.execute(select_sql, {'chunk_name': chunk_name}) 
    row = cursor.fetchone()
    if row:
        chunk_data = {
            'chunk_name': row['chunk_name'],
            'namespace': row['namespace'],
            'chunk_archive_path': row['chunk_archive_path'],
            'chunk_extracted_path': row['chunk_extracted_path'],
            'downloaded_at': row['downloaded_at'],
            'completed_at': row['completed_at']
        }
    else:
        logger.warning(f"No chunk found with chunk_name: {chunk_name}")
        chunk_data = {}
    return chunk_data

def update_embeddings_for_page(page_data: dict, sqlconn: sqlite3.Connection) -> None:
    # convert numpy arrays to bytes for storage
    embedding_vector_bytes = np.array(page_data['embedding_vector']).astype(np.float32).tobytes() if isinstance(page_data['embedding_vector'], np.ndarray) else page_data['embedding_vector']
    reduced_vector_bytes = np.array(page_data['reduced_vector']).astype(np.float32).tobytes() if isinstance(page_data['reduced_vector'], np.ndarray) else page_data['reduced_vector']
    three_d_vector_text = json.dumps(page_data['three_d_vector']) if page_data.get('three_d_vector') else None
    prepared_data = {
        key: embedding_vector_bytes if key == 'embedding_vector' else 
             reduced_vector_bytes if key == 'reduced_vector' else 
             three_d_vector_text if key == 'three_d_vector' else 
             value
        for key, value in page_data.items()
    }

    logger.debug(f"Updating page embedding: {page_data['page_id']}")
    
    update_page_vector_sql = """
            UPDATE page_log 
            SET embedding_vector = :embedding_vector,
                reduced_vector = :reduced_vector,
                three_d_vector = :three_d_vector
            WHERE page_id = :page_id
            """
    
    try:
        cursor = sqlconn.cursor()
        cursor.execute(update_page_vector_sql, prepared_data)
        sqlconn.commit()
    except sqlite3.Error as e:
        try:
            sqlconn.rollback()
        except Exception as e:
            logger.error(f"Failed to roll back sql transaction while handling another error: {e}")
            pass
        logger.exception(f"Failed to update page embedding for page {page_data['page_id']}: {e}")
        raise

    #np.frombuffer(blob_data, dtype=np.float32)

def get_page_ids_needing_embedding_for_chunk(chunk_name: str, sqlconn: sqlite3.Connection) -> list[int]:
    select_sql = """
        SELECT page_id 
        FROM page_log
        WHERE chunk_name = :chunk_name
        AND embedding_vector IS NULL
        ORDER BY page_id ASC;
    """
    cursor = sqlconn.execute(select_sql, {'chunk_name': chunk_name})
    rows = cursor.fetchall()    
    page_id_list = [row['page_id'] for row in rows]
    return page_id_list
