
# Dimensionality Reduction & Clustering Implementation Plan

## Overview
This document details the step‑by‑step plan to extend the **wp‑embeddings** project with the following pipeline:

1. **PCA** – Reduce the original embedding vectors to 100 dimensions.  
2. **K‑Means** – Cluster the PCA‑reduced vectors into 100 clusters.  
3. **UMAP** – Within each cluster, further reduce vectors to 3‑D space.  

All derived data will be persisted in the SQLite database, enabling downstream queries such as:
- List of clusters with their 3‑D centroids.  
- Cluster assignment for each page.  
- 3‑D vector for each page within its cluster.

## 1. Database Schema Changes

### 1.1 New Table: `page_vector`
```sql
CREATE TABLE IF NOT EXISTS page_vector (
    page_id INTEGER PRIMARY KEY REFERENCES page_log(page_id) ON DELETE CASCADE,
    embedding_vector BLOB,    -- 2048-dimensional numpy array that is returned by the embedding function
    reduced_vector BLOB,      -- float32 array, 100‑dimensional
    cluster_id INTEGER,       -- FK to cluster_info.cluster_id
    three_d_vector TEXT       -- JSON array "[x, y, z]"
);
```

### 1.2 New Table: `cluster_info`
```sql
CREATE TABLE IF NOT EXISTS cluster_info (
    cluster_id INTEGER PRIMARY KEY,
    centroid_3d TEXT,         -- JSON array "[x, y, z]"
    cluster_name TEXT,        -- to be generated
    cluster_description TEXT  -- to be generated
);
```

### 1.3 Migration Steps
1. **Create tables** if they do not exist (run on startup via `ensure_tables` in `database.py`).  
2. **Back‑fill** existing pages:  
   - For each row in `page_log` that still contains `embedding_vector`, copy it into `page_vector` (populate `reduced_vector` and `three_d_vector` as `NULL`).  
   - Remove the three currently existing columns (`embedding_vector`, `reduced_vector`, `three_d_vector`) from `page_log` after migration.  
3. **Add index** on `page_vector.cluster_id` for fast look‑ups.

## 2. Utility Module `transform.py`

### 2.1 Functions
| Function | Description | File Target |
|----------|-------------|-------------|
| `run_pca(sqlconn, target_dim=100)` | Loads all `embedding_vector` blobs, fits `sklearn.decomposition.PCA`, stores the 100‑dim vectors in `page_vector.reduced_vector`. This must work even if all embedding vectors do not fit into memory. Consider using generators and streaming data through the PCA process if that is an option with sklearn. | add to new file `transform.py` |
| `run_kmeans(sqlconn, n_clusters=100)` | Reads `reduced_vector` from `page_vector`, fits `sklearn.cluster.KMeans`, writes `cluster_id` to `page_vector` and populates `cluster_info` (initial `centroid_3d` = `NULL`). | add to new file `transform.py` |
| `run_umap_per_cluster(sqlconn, n_components=3, limit=1)` | For each cluster, loads its `reduced_vector`s, fits `umap.UMAP`, stores the 3‑D vectors as JSON in `page_vector.umap_3d_vector`. Updates `cluster_info.centroid_3d` with the mean of the 3‑D vectors. do the number of clusters specified by `limit`. | add to new file `transform.py` |
| `fetch_page_vectors(sqlconn, columns)` | Helper to retrieve vectors for a given set of columns (used by the above functions). | add to file `database.py` |
| `store_blob(sqlconn, page_id, column, np_array)` | Serialises a NumPy array to `float32` bytes and updates the specified column. |  add to file `database.py` |
| `store_json(sqlconn, page_id, column, data)` | Serialises Python data to JSON string and updates the column. |  add to file `database.py` |

### 2.2 Dependencies
- `numpy`
- `scikit-learn`
- `umap-learn`
- `json` (standard library)

Run `uv add numpy scikit-learn umap-learn` to add them to the project.

## 3. Update `database.py`

1. **Add import** for the new tables in `ensure_tables`.  
2. **Create helper functions**:
   - `insert_page_vector(page_id, sqlconn)` – inserts a row with `NULL` placeholders if not present.  
   - `update_reduced_vector(page_id, vector, sqlconn)`  
   - `update_cluster_id(page_id, cluster_id, sqlconn)`  
   - `update_umap_vector(page_id, vector, sqlconn)`  
   - `upsert_cluster_info(cluster_id, centroid, count, sqlconn)`  
3. **Modify existing fetchers** (`get_page_by_id`, `get_any_page`) to join with `page_vector` when needed.  
4. **Remove** references to the old columns (`embedding_vector`, `reduced_vector`, `three_d_vector`) from any function that still accesses them.

## 4. New Command Class `ReduceCommand`

File: `command.py` (or a new submodule `commands/transform.py`).

```python
class ReduceCommand(Command):
    """Run PCA on all pages."""
    def __init__(self):
        super().__init__(
            name="reduce",
            description="Perform principal component analysis dimension reduction on page embeddings",
            required_args=[],
            optional_args={
                "pca_dim": 100
            }
        )

    def execute(self, args):
        sqlconn = get_sql_conn()
        ensure_tables(sqlconn)

        pca_dim = int(args.get("pca_dim", 100))

        logger.info("Starting PCA (target_dim=%d)", pca_dim)
        run_pca(sqlconn, target_dim=pca_dim)

        return "✓ Principal component analysis dimension reduction completed."
```

### Registration
In `CommandInterpreter._register_commands()` add:
```python
self.parser.register_command(ReduceCommand())
```


## 5. New Command Class `ClusterCommand`

File: `command.py` (or a new submodule `commands/transform.py`).

```python
class ClusterCommand(Command):
    """Run K‑Means on all (or some) pages."""
    def __init__(self):
        super().__init__(
            name="cluster",
            description="Perform clustering on PCA reduced page vectors",
            required_args=[],
            optional_args={
                "clusters": 100
            }
        )

    def execute(self, args):
        sqlconn = get_sql_conn()
        ensure_tables(sqlconn)

        n_clusters = int(args.get("clusters", 100))

        logger.info("Starting K‑Means (n_clusters=%d)", n_clusters)
        run_kmeans(sqlconn, n_clusters=n_clusters)

        return "✓ Clustering completed."
```

### Registration
In `CommandInterpreter._register_commands()` add:
```python
self.parser.register_command(ClusterCommand())
```


## 4. New Command Class `Map3DCommand`

File: `command.py` (or a new submodule `commands/transform.py`).

```python
class Map3dCommand(Command):
    """Run UMAP on each cluster."""
    def __init__(self):
        super().__init__(
            name="map3d",
            description="Map each page cluster to 3-space using UMAP on the PCA reduced vectors",
            required_args=[],
            optional_args={
                "umap_dim": 3,
                "limit": 1
            }
        )

    def execute(self, args):
        sqlconn = get_sql_conn()
        ensure_tables(sqlconn)

        limit = int(args.get("limit", 1))
        umap_dim = int(args.get("umap_dim", 3))

        logger.info("Starting UMAP (n_components=%d)", umap_dim)
        completed_cluster_count = run_umap_per_cluster(sqlconn, n_components=umap_dim, limit=limit)

        return f"✓ Mapping clusters to 3-space completed. Mapped {completed_cluster_count} clusters."
```

### Registration
In `CommandInterpreter._register_commands()` add:
```python
self.parser.register_command(Map3DCommand())
```

## 7. CLI Documentation & README Updates

- Add a **Usage** section in `README.md` showing:
  ```bash
  python command.py process_vectors --pca_dim 100 --clusters 100 --umap_dim 3
  ```
- Document the new tables and their purpose.
- Explain how to query results, e.g.:
  ```sql
  SELECT c.cluster_id, c.centroid_3d, p.page_id, p.umap_3d_vector
  FROM cluster_info c
  JOIN page_vector p ON c.cluster_id = p.cluster_id;
  ```

## 8. Testing Strategy

1. **Unit Tests** for each function in `transform.py` using small synthetic datasets (e.g., 10 pages, 5‑dim vectors).  
2. **Integration Test** that runs the full pipeline on a temporary in‑memory SQLite DB and asserts:
   - `page_vector` rows exist for every page.  
   - `cluster_info` has exactly `n_clusters` rows.  
   - Each `umap_3d_vector` is a JSON array of length 3.  
3. **Performance Test** (optional) on a subset of real data to ensure runtime is acceptable.

## 9. Migration & Backward Compatibility

- The migration script will **preserve existing data** by copying `embedding_vector` into the new `page_vector` table before dropping the old columns.  
- Existing commands (`embed`, `status`, etc.) will continue to work because they now reference the new helpers in `database.py`.  
- No external API changes are required.

## 9. Risk Assessment & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large memory usage when loading all embeddings for PCA | High (OOM) | Process embeddings in batches; use `IncrementalPCA` if needed. |
| Cluster count mismatch with sparse data | Medium | Validate that each cluster receives at least one page; fallback to fewer clusters if necessary. |
| Migration failure (data loss) | High | Perform migration inside a transaction; keep a backup of the original DB file. |
| Dependency version conflicts | Low | Pin versions in `pyproject.toml` and run tests in a clean virtual environment. |

---

**Next Steps**

1. Add the new tables to `ensure_tables` (code change).  
2. Implement `transform.py` with the functions listed above.  
3. Update `database.py` with the new helper methods.  
4. Create the `TransformCommand` and register it.  
5. Write migration script and run it on a test DB.  
6. Update documentation and add tests.

Please review this plan and let me know if any adjustments are needed before we proceed to implementation.