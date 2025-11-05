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

# Cluster Tree Visualizations
# ---------------------------
# Create traditional dendrogram (recommended for clarity)
python visualize_embeddings.py --traditional-dendrogram --max-nodes 50

# Create hierarchical visualization with multiple layouts
python visualize_embeddings.py --hierarchical --max-nodes 30

# Create radial dendrogram (improved version)
python visualize_embeddings.py --radial-dendrogram --max-nodes 100

# List cluster tree information
python visualize_embeddings.py --list-tree

# Analyze cluster tree data quality and issues
python visualize_embeddings.py --analyze-tree

# Validate cluster tree integrity
python visualize_embeddings.py --validate-tree

# Advanced options for tree visualizations
python visualize_embeddings.py --traditional-dendrogram --namespace enwiki_namespace_0 --max-depth 5 --max-nodes 25 --save-path dendrogram.png
```

## Database Schema

The visualization tools query the following tables:

### Embedding Tables
- `page_vector`: Contains the 3D embeddings (`three_d_vector` column)
- `page_log`: Contains page titles and URLs
- `chunk_log`: Contains namespace information

### Cluster Tree Tables
- `cluster_tree`: Contains hierarchical cluster structure with parent-child relationships
  - `node_id`: Unique identifier for each cluster node
  - `parent_id`: Reference to parent node (NULL for root nodes)
  - `depth`: Depth level in the hierarchy
  - `doc_count`: Number of documents in this cluster
  - `child_count`: Number of direct children
  - `first_label`/`final_label`: Labels for the cluster

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

### Cluster Tree Visualizations
- **Traditional Dendrogram**: Clear hierarchical view using scipy's dendrogram function
- **Hierarchical Visualization**: Four different layout approaches:
  - Traditional top-down dendrogram
  - Level-by-level visualization
  - Rectangular tree layout (better for large trees)
  - Improved radial layout (only for small trees ≤25 nodes)
- **Radial Dendrogram**: Circular layout showing hierarchical relationships

## Tips for Exploration

### For Cluster Analysis
1. **Start with small clusters**: Begin with clusters that have fewer pages to get a feel for the data
2. **Look for patterns**: Notice how articles about similar topics group together
3. **Compare clusters**: Different clusters may represent different topics or themes
4. **Use both 2D and 3D**: 2D is faster, 3D shows more detail
5. **Adjust limits**: Use the `--limit` parameter to control how many pages are displayed

### For Cluster Tree Analysis
1. **Use traditional dendrogram for clarity**: The traditional dendrogram provides the clearest view of hierarchical relationships
2. **Start with hierarchical visualization**: Shows multiple layout approaches to understand your data structure
3. **Validate data quality**: Use `--analyze-tree` and `--validate-tree` to check for data issues
4. **Limit node count**: Use `--max-nodes` to control visualization complexity (25-100 nodes works well)
5. **Filter by depth**: Use `--max-depth` to focus on specific hierarchy levels

## Example Workflow

### Standard Cluster Analysis
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

### Cluster Tree Analysis
```bash
# 1. Check if cluster tree exists
python visualize_embeddings.py --list-tree

# 2. Analyze data quality and potential issues
python visualize_embeddings.py --analyze-tree

# 3. Validate tree structure integrity
python visualize_embeddings.py --validate-tree

# 4. Create traditional dendrogram (recommended for clarity)
python visualize_embeddings.py --traditional-dendrogram --max-nodes 50

# 5. Try hierarchical visualization for multiple perspectives
python visualize_embeddings.py --hierarchical --max-nodes 30

# 6. For small trees, try radial layout
python visualize_embeddings.py --radial-dendrogram --max-nodes 25
```

### Advanced Analysis Workflow
```bash
# 1. Comprehensive tree analysis
python visualize_embeddings.py --analyze-tree --namespace enwiki_namespace_0

# 2. Validate integrity and check for issues
python visualize_embeddings.py --validate-tree --namespace enwiki_namespace_0

# 3. Create traditional dendrogram with depth filtering
python visualize_embeddings.py --traditional-dendrogram --max-depth 5 --max-nodes 100

# 4. Save visualization for documentation
python visualize_embeddings.py --hierarchical --max-nodes 40 --save-path cluster_analysis.png

# 5. Interactive exploration
python visualize_embeddings.py --interactive
#   Then use commands: 'tree', 'analyze', 'validate', 'traditional', 'hierarchical'
```

## Dependencies

### Core Dependencies
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` for 2D/3D plotting (script version)
- `plotly` for interactive visualizations (notebook version)
- `scikit-learn` for PCA analysis
- `sqlite3` (built-in Python module)

### Additional Dependencies for Tree Visualizations
- `scipy` for hierarchical clustering and dendrogram functions
  - Required for: `--traditional-dendrogram`, `--hierarchical`, `--analyze-tree`, `--validate-tree`

These should already be installed as part of the wp-embeddings project dependencies. If scipy is missing, install it with:
```bash
pip install scipy
```

## Troubleshooting

### Database Connection Issues
Make sure you're in the correct directory and the `chunk_log.db` file exists.

### Missing Dependencies
Install missing packages with:
```bash
pip install pandas numpy matplotlib plotly scikit-learn scipy
```

### Empty Results
If no clusters are found, you may need to run the clustering step first:
```bash
python -m command cluster --namespace enwiki_namespace_0 --clusters 100
```

### Cluster Tree Issues
If cluster tree commands fail with "No cluster tree nodes found":
1. Check if the `cluster_tree` table exists in your database
2. You may need to generate cluster tree data first (check project documentation)
3. Try a different namespace if you're using a custom one

### Visualization Problems
- **"Too many nodes" warning**: Use `--max-nodes` to limit the number of displayed nodes
- **Poor dendrogram layout**: Try `--traditional-dendrogram` instead of radial for better clarity
- **Overlapping labels**: Reduce `--max-nodes` or use `--max-depth` to filter the hierarchy
- **Missing data**: Run `--analyze-tree` to check for data quality issues

### Performance Issues
- For large datasets (>1000 nodes), use `--max-nodes 50` or less
- Use `--max-depth` to focus on specific hierarchy levels
- Consider using 2D visualizations (`--2d`) instead of 3D for better performance

## Contributing

If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## New Features in Recent Updates

### Enhanced Dendrogram Visualizations
- **Traditional Dendrogram**: Added a clear, hierarchical dendrogram using scipy's built-in functions
- **Hierarchical Visualization**: Four different layout approaches for better data understanding
- **Improved Radial Layout**: Better spacing and algorithms for small trees

### Data Quality and Validation Tools
- **Tree Quality Analysis** (`--analyze-tree`): Detects cycles, connectivity issues, depth inconsistencies, and data quality problems
- **Integrity Validation** (`--validate-tree`): Validates parent references, depth consistency, and tree structure
- **Enhanced Error Handling**: Better detection and reporting of data issues

### Interactive Mode Enhancements
- New commands: `traditional`, `analyze`, `validate`
- Better help system with examples
- Improved error handling and user feedback

### Performance Optimizations
- Node limiting with `--max-nodes` parameter
- Depth filtering with `--max-depth` parameter
- Better memory management for large datasets
- Improved layout algorithms for clearer visualizations