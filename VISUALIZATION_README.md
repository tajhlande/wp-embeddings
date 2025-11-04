# Wikipedia Embeddings Visualization Tools

This directory contains tools for visualizing the 3D embeddings of Wikipedia pages stored in your SQLite database.

## Overview

The wp-embeddings project processes Wikipedia articles through several stages:
1. Download and extract article content
2. Compute embeddings (vector representations)
3. Reduce dimensionality using PCA
4. Cluster pages using k-means
5. Project clusters into 3D space using UMAP

These visualization tools help you understand the results of this processing by showing how similar Wikipedia articles are grouped together in the embedding space.

## Files

### 1. `visualize_embeddings.ipynb` (Jupyter Notebook)
An interactive Jupyter notebook with rich visualizations using Plotly.

**Features:**
- Interactive 3D scatter plots with zoom, rotate, and pan capabilities
- Dropdown menus to select specific clusters
- Page title annotations (instead of numeric IDs)
- 2D PCA projections for easier interpretation
- Progress tracking across all clusters
- Hover tooltips showing page titles and URLs

**Usage:**
```bash
jupyter notebook visualize_embeddings.ipynb
```

### 2. `visualize_embeddings.py` (Python Script)
A command-line script for creating static visualizations using matplotlib.

**Usage:**
```bash
# List all clusters
python visualize_embeddings.py --list-clusters

# Show cluster statistics
python visualize_embeddings.py --stats

# Visualize a specific cluster (3D)
python visualize_embeddings.py --cluster 5 --limit 100

# Visualize a specific cluster (2D)
python visualize_embeddings.py --cluster 5 --2d --limit 200

# Start interactive mode
python visualize_embeddings.py --interactive
```

## Database Schema

The visualization tools query the following tables:

- `page_vector`: Contains the 3D embeddings (`three_d_vector` column)
- `page_log`: Contains page titles and URLs
- `cluster_info`: Contains cluster metadata
- `chunk_log`: Contains namespace information

## Common Issues and Solutions

### 1. No 3D vectors available
If you see "No projected vectors found" messages, it means you need to run the projection step:

```bash
python -m command project --namespace enwiki_namespace_0 --limit 10
```

### 2. Cluster IDs not matching
The visualization tools use the `cluster_id` from the `page_vector` table, which should match the clusters created during the clustering step.

### 3. Performance issues with large clusters
For clusters with many pages (>500), consider using the `--limit` parameter to visualize a subset of pages.

## Understanding the Visualizations

### 3D Scatter Plots
- Each point represents a Wikipedia article
- Position in 3D space reflects the article's content similarity
- Points that are close together represent similar topics
- Color indicates cluster membership
- Hover over points to see article titles

### 2D PCA Projections
- 2D representation of the 3D data using Principal Component Analysis
- Shows the main patterns in the data while losing some detail
- Faster to render and easier to interpret than 3D plots
- Explained variance percentage shows how much information is preserved

### Cluster Statistics
- **Total Pages**: Number of articles in the cluster
- **Projected**: Number of articles with 3D vectors computed
- **Complete**: Shows projection progress (✓ when 100% complete)

## Tips for Exploration

1. **Start with small clusters**: Begin with clusters that have fewer pages to get a feel for the data
2. **Look for patterns**: Notice how articles about similar topics group together
3. **Compare clusters**: Different clusters may represent different topics or themes
4. **Use both 2D and 3D**: 2D is faster, 3D shows more detail
5. **Adjust limits**: Use the `--limit` parameter to control how many pages are displayed

## Example Workflow

```bash
# 1. Check current status
python visualize_embeddings.py --list-clusters

# 2. Show detailed statistics
python visualize_embeddings.py --stats

# 3. Visualize a cluster with few pages
python visualize_embeddings.py --cluster 0 --limit 50

# 4. If needed, project more clusters
python -m command project --namespace enwiki_namespace_0 --limit 5

# 5. Explore in interactive mode
python visualize_embeddings.py --interactive
```

## Dependencies

The visualization tools require the following Python packages:

- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` for 2D/3D plotting (script version)
- `plotly` for interactive visualizations (notebook version)
- `scikit-learn` for PCA analysis
- `sqlite3` (built-in Python module)

These should already be installed as part of the wp-embeddings project dependencies.

## Troubleshooting

### Database Connection Issues
Make sure you're in the correct directory and the `chunk_log.db` file exists.

### Missing Dependencies
Install missing packages with:
```bash
pip install pandas numpy matplotlib plotly scikit-learn
```

### Empty Results
If no clusters are found, you may need to run the clustering step first:
```bash
python -m command cluster --namespace enwiki_namespace_0 --clusters 100
```

## Contributing

If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.