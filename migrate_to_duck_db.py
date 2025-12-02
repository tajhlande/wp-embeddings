import duckdb
import sqlite3

import pandas as pd

from database import bytes_to_numpy

sqlite_conn = sqlite3.connect('chunk_log.db')
sqlite_conn.row_factory = sqlite3.Row
duckconn = duckdb.connect("wp-embeddings.duckdb")

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

page_log_table_sql = """
    CREATE TABLE IF NOT EXISTS page_log (
        namespace TEXT NOT NULL,
        page_id INTEGER NOT NULL,
        title TEXT,
        chunk_name TEXT,
        url TEXT,
        extracted_at DATETIME,
        abstract TEXT,
        PRIMARY KEY (namespace, page_id),
        FOREIGN KEY (namespace, chunk_name) REFERENCES chunk_log (namespace, chunk_name)
    );
    """

# New tables for vector storage and clustering information
page_vector_table_sql = """
    CREATE TABLE IF NOT EXISTS page_vector (
        namespace TEXT NOT NULL,
        page_id INTEGER NOT NULL,
        embedding_vector FLOAT[],    -- original 2048-dim numpy array (float32)
        reduced_vector FLOAT[],      -- PCA-reduced 100-dim array (float32)
        cluster_id INTEGER,       -- FK to cluster_info.cluster_id
        cluster_node_id INTEGER,  -- FK to cluster_tree.node_id
        three_d_vector JSON,       -- JSON "[x, y, z]"
        PRIMARY KEY (namespace, page_id),
        FOREIGN KEY (namespace, page_id) REFERENCES page_log (namespace, page_id)
    );
    """

cluster_tree_table_sql = """
    CREATE TABLE IF NOT EXISTS cluster_tree (
        namespace TEXT,
        node_id BIGINT ,
        parent_id BIGINT,
        depth SMALLINT NOT NULL,
        centroid FLOAT[],
        doc_count INT NOT NULL,
        top_terms TEXT,
        sample_doc_ids TEXT,
        child_count INT DEFAULT 0,
        first_label TEXT,
        final_label TEXT,
        PRIMARY KEY (namespace, node_id)
    );
    """

print("Building SQL schema")
duckconn.execute(chunk_log_table_sql)
duckconn.execute(page_log_table_sql)
duckconn.execute(page_vector_table_sql)
duckconn.execute(cluster_tree_table_sql)


chunk_log_migrate_sql = """
INSERT INTO chunk_log
SELECT
    namespace,
    chunk_name,
    chunk_archive_path,
    chunk_extracted_path,
    downloaded_at,
    unpacked_at
FROM sqlite_scan('chunk_log.db', 'chunk_log');
"""

print("Migrating chunk_log")
duckconn.execute(chunk_log_migrate_sql)

page_log_migrate_sql = """
INSERT INTO page_log
SELECT
    namespace,
    page_id,
    title,
    chunk_name,
    url,
    extracted_at,
    abstract
FROM sqlite_scan('chunk_log.db', 'page_log');
"""

print("Migrating page_log")
duckconn.execute(page_log_migrate_sql)

print("Migrating page_vector")
chunk_counter = 0
for chunk in pd.read_sql_query("SELECT * FROM page_vector", sqlite_conn, chunksize=50000):
    chunk_counter += 1
    print(f"    chunk {chunk_counter}")
    chunk['embedding_vector'] = chunk['embedding_vector'].apply(bytes_to_numpy)  # type: ignore
    chunk['reduced_vector'] = chunk['reduced_vector'].apply(bytes_to_numpy)  # type: ignore
    duckconn.execute("INSERT INTO page_vector SELECT * FROM chunk")


print("Migrating cluster info")
chunk_counter = 0
for chunk in pd.read_sql_query("SELECT * FROM cluster_info", sqlite_conn, chunksize=50000):
    chunk_counter += 1
    print(f"    chunk {chunk_counter}")
    chunk['centroid_vector'] = chunk['centroid_vector'].apply(bytes_to_numpy)  # type: ignore
    duckconn.execute("INSERT INTO cluster_info SELECT * FROM chunk")

print("Migrating cluster tree")
chunk_counter = 0
for chunk in pd.read_sql_query("SELECT * FROM cluster_tree", sqlite_conn, chunksize=50000):
    chunk_counter += 1
    print(f"    chunk {chunk_counter}")
    chunk['centroid'] = chunk['centroid'].apply(bytes_to_numpy)  # type: ignore
    duckconn.execute("INSERT INTO cluster_tree SELECT * FROM chunk")


print("Completed migration")
