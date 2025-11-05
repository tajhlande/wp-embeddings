# Transformation notes

Notes on specific transformation algorithms are to be recorded here

## Recursive clustering algorithm
This algorithm recursively clusters the entire corpus of reduced vectors
to produce a tree of clusters.

**Storage:**
A database table called `cluster_tree` that contains the necessary data about cluster tree structure.
Define this table in the `database.py` file at the top in the `ensure_tables()` function where the other tables are defined.

| Column         | Type                                  | Meaning                                    |
|----------------|---------------------------------------|--------------------------------------------|
| namespace      | TEXT PRIMARY KEY                      | Each cluster tree belongs to one namespace |
| node_id        | BIGINT PRIMARY KEY                    | Unique identifier within namespace         |
| parent_id      | BIGINT FOREIGN KEY REFERENCES node_id | Null for root                              |
| depth          | SMALLINT                              | 0‑based depth                              |
| centroid       | BLOB                                  | 100‑D vector (or 20‑D after reduction)     |
| doc_count      | INT                                   | How many docs belong here                  |
| top_terms      | TEXT (JSON)                           | ["machine learning","neural network",...]  |
| sample_doc_ids | TEXT (JSON)                           | [12345, 67890,…]                           |
| child_count    | INT                                   | Denormalised for quick UI decisions        |

**Indexing:**
- Primary index on `node_id.`
- Secondary index on `(parent_id, depth)`.
- Full‑text index on `top_terms` for search.



**Algorithm steps:**
1. Start with one node that contains all documents.
    * `cluster_size` = N = 7 018 194 if that's how many documents we have to start.
    * Compute `k = ceil(N / leaf_target)`.
    * If `leaf_target = 50`, then `k = ceil(7 018 194 / 50) = 140 364`.
    * That number is far too large for a practical split, so you cap it (e.g., `max_k = 50`). The root will therefore be split into at most 50 children.

2.  Run the chosen clustering algorithm (Mini‑Batch K‑Means with the capped `k`).
    * The algorithm returns `k` clusters, each with its own set of document IDs.
    * For eac child you now know its `cluster_size` (just the length of the ID list it received).
    * Alternative clustering methods: bisecting‑K‑Means, HDBSCAN.  But let's use Mini-Batch K-Means for now.

3.  Recurse on every child:
    * For a child that received, say, 180 000 documents, `cluster_size = 180 000`.
    * Compute its own `k_child = ceil(180 000 / 50)` = `3 600`.
    * Again cap to the same `max_k` (or a depth‑dependent cap).
    * Split that child into at most `max_k` sub‑clusters, obtain their sizes, and repeat.

4.  Stopping condition – you become a leaf when any of the following holds:
   * `cluster_size ≤ leaf_target` (the node is already small enough).
   * `depth == max_depth` (you have reached a depth you consider reasonable for the UI).
   * The clustering quality is poor (e.g., silhouette < 0.1 from `sklearn.metrics.silhouette_score`). In that case you may also decide to stop early even if the size is larger than `leaf_target`.

5. Record the node information into the database table `cluster_tree` by giving each node (cluster or leaf) an ID and having a reference to the parent node's ID