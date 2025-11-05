#!/usr/bin/env python3
"""
Wikipedia Embeddings Visualization Tool

This script provides command-line and interactive visualization capabilities for
the 3D embeddings stored in the SQLite database.

Usage:
    python visualize_embeddings.py --cluster <cluster_id> [--limit <n>] [--2d]
    python visualize_embeddings.py --list-clusters
    python visualize_embeddings.py --interactive
    python visualize_embeddings.py --radial-dendrogram [--namespace <namespace>] [--max-depth <n>] [--max-nodes <n>] [--save-path <path>]
    python visualize_embeddings.py --traditional-dendrogram [--namespace <namespace>] [--max-depth <n>] [--max-nodes <n>] [--save-path <path>]
    python visualize_embeddings.py --hierarchical [--namespace <namespace>] [--max-depth <n>] [--max-nodes <n>] [--save-path <path>]
    python visualize_embeddings.py --list-tree [--namespace <namespace>]
"""

import argparse
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Optional, List, Dict, Tuple
import logging
from collections import defaultdict

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """Main class for visualizing Wikipedia embeddings."""

    def __init__(self, db_path: str = "chunk_log.db"):
        self.db_path = db_path
        self.conn = None

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_cluster_info(self, namespace: str = 'enwiki_namespace_0') -> pd.DataFrame:
        """Get information about all clusters."""
        logger.debug(f"Getting cluster info for namespace: {namespace}")
        conn = self.get_connection()

        query = """
            SELECT
                cluster_id,
                COUNT(*) as page_count,
                SUM(CASE WHEN three_d_vector IS NOT NULL THEN 1 ELSE 0 END) as projected_count
            FROM page_vector
            INNER JOIN page_log ON page_vector.page_id = page_log.page_id
            INNER JOIN chunk_log ON page_log.chunk_name = chunk_log.chunk_name
            WHERE chunk_log.namespace = ?
            AND cluster_id IS NOT NULL
            GROUP BY cluster_id
            ORDER BY cluster_id
        """

        try:
            df = pd.read_sql_query(query, conn, params=(namespace,))
            logger.debug(f"Found {len(df)} clusters")
            return df
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            raise

    def load_embeddings_for_cluster(self, cluster_id: int, namespace: str = 'enwiki_namespace_0',
                                    limit: Optional[int] = None) -> pd.DataFrame:
        """Load 3D embeddings and page information for a specific cluster."""
        conn = self.get_connection()

        query = """
            SELECT
                pv.page_id,
                pv.three_d_vector,
                pl.title,
                pl.url,
                pv.cluster_id
            FROM page_vector pv
            INNER JOIN page_log pl ON pv.page_id = pl.page_id
            INNER JOIN chunk_log cl ON pl.chunk_name = cl.chunk_name
            WHERE pv.cluster_id = ?
            AND cl.namespace = ?
            AND pv.three_d_vector IS NOT NULL
            ORDER BY pv.page_id
        """

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=(cluster_id, namespace))

        # Parse the 3D vector JSON
        def parse_vector(vector_str):
            try:
                if pd.isna(vector_str) or vector_str is None:
                    logger.warning("Null vector encountered")
                    return np.array([0, 0, 0])
                vector_data = json.loads(vector_str)
                if not isinstance(vector_data, list) or len(vector_data) != 3:
                    logger.warning(f"Invalid vector format: {vector_str}")
                    return np.array([0, 0, 0])
                return np.array(vector_data)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error for vector: {e}, vector_str: {vector_str}")
                return np.array([0, 0, 0])
            except Exception as e:
                logger.warning(f"Unexpected error parsing vector: {e}, vector_str: {vector_str}")
                return np.array([0, 0, 0])

        # Parse vectors and extract coordinates
        vectors = []
        for vector_str in df['three_d_vector']:
            try:
                if pd.isna(vector_str) or vector_str is None:
                    vectors.append([0, 0, 0])
                    continue
                vector_data = json.loads(vector_str)
                if not isinstance(vector_data, list) or len(vector_data) != 3:
                    vectors.append([0, 0, 0])
                else:
                    vectors.append(vector_data)
            except (json.JSONDecodeError, Exception):
                vectors.append([0, 0, 0])

        vectors_array = np.array(vectors)
        df['x'] = vectors_array[:, 0]
        df['y'] = vectors_array[:, 1]
        df['z'] = vectors_array[:, 2]

        logger.debug(f"Loaded {len(df)} embeddings for cluster {cluster_id}")
        return df

    def plot_3d_cluster(self, cluster_id: int, limit: int = 100, save_path: Optional[str] = None):
        """Create a 3D visualization of a cluster."""
        df = self.load_embeddings_for_cluster(cluster_id, limit=limit)

        if len(df) == 0:
            print(f"No projected vectors found for cluster {cluster_id}")
            return

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore

        # Create scatter plot
        scatter = ax.scatter(df['x'], df['y'], df['z'],
                             c=df['cluster_id'], cmap='viridis',
                             alpha=0.6, marker='o')

        # Add labels for first 20 points to avoid clutter
        for i, row in df.head(30).iterrows():
            text_label = f"{row['title'][:30]}..."
            ax.text(row['x'], row['y'], row['z'], s=text_label,  # type: ignore
                    fontsize=8, alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z', fontsize=10)  # type: ignore
        ax.set_title(f'3D Visualization of Cluster {cluster_id} ({len(df)} pages)')

        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_2d_cluster(self, cluster_id: int, limit: int = 200, save_path: Optional[str] = None):
        """Create a 2D visualization using PCA."""
        df = self.load_embeddings_for_cluster(cluster_id, limit=limit)

        if len(df) == 0:
            print(f"No projected vectors found for cluster {cluster_id}")
            return

        # Extract 3D vectors and apply PCA to reduce to 2D
        vectors = np.array(df[['x', 'y', 'z']])
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        df['pca_x'] = vectors_2d[:, 0]
        df['pca_y'] = vectors_2d[:, 1]

        # Create 2D scatter plot
        plt.figure(figsize=(12, 10))

        # Plot all points
        plt.scatter(df['pca_x'], df['pca_y'], alpha=0.6, s=30)

        # Add labels for first 30 points
        for i, row in df.head(30).iterrows():
            plt.annotate(f"{row['title'][:25]}...",
                         (row['pca_x'], row['pca_y']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'2D PCA Projection of Cluster {cluster_id} ({len(df)} pages)')
        plt.grid(True, alpha=0.3)

        # Add explained variance info
        explained_var = pca.explained_variance_ratio_
        plt.figtext(0.02, 0.02,
                    f'Explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def list_clusters(self):
        """List all clusters with their statistics."""
        cluster_info = self.get_cluster_info()

        print("\n=== Cluster Information ===")
        print(f"{'Cluster ID':<12} {'Total Pages':<12} {'Projected':<12} {'Complete':<10}")
        print("-" * 50)

        for _, row in cluster_info.iterrows():
            complete = "✓" if row['projected_count'] == row['page_count'] else \
                      f"{row['projected_count']}/{row['page_count']}"
            print(f"{row['cluster_id']:<12} {row['page_count']:<12} {row['projected_count']:<12} {complete:<10}")

        total_pages = cluster_info['page_count'].sum()
        total_projected = cluster_info['projected_count'].sum()
        overall_percentage = (total_projected / total_pages * 100) if total_pages > 0 else 0

        print("-" * 50)
        print(f"Total: {total_pages} pages, {total_projected} projected ({overall_percentage:.1f}%)")

    def show_cluster_statistics(self):
        """Show detailed cluster statistics."""
        cluster_info = self.get_cluster_info()

        # Create a simple text-based progress bar
        print("\n=== Projection Progress ===")
        for _, row in cluster_info.iterrows():
            percentage = (row['projected_count'] / row['page_count'] * 100) if row['page_count'] > 0 else 0
            bar_length = 30
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)

            print(f"Cluster {row['cluster_id']:2d}: |{bar}| {percentage:5.1f}% "
                  f"({row['projected_count']}/{row['page_count']})")

    def interactive_mode(self):
        """Start interactive mode."""
        print("=== Interactive Embedding Visualization ===")
        print("Type 'help' for commands, 'quit' to exit")

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'list':
                    self.list_clusters()
                elif command == 'stats':
                    self.show_cluster_statistics()
                elif command == 'tree':
                    self.list_cluster_tree_info()
                elif command == 'radial':
                    self.plot_radial_dendrogram()
                elif command == 'traditional':
                    self.plot_traditional_dendrogram()
                elif command == 'analyze':
                    self.analyze_cluster_tree_quality()
                elif command == 'validate':
                    self.validate_cluster_tree_integrity()
                elif command.startswith('plot '):
                    parts = command.split()
                    if len(parts) >= 2:
                        try:
                            cluster_id = int(parts[1])
                            limit = int(parts[2]) if len(parts) > 2 else 100
                            self.plot_3d_cluster(cluster_id, limit)
                        except ValueError:
                            print("Invalid cluster ID or limit")
                    else:
                        print("Usage: plot <cluster_id> [limit]")
                elif command.startswith('plot2d '):
                    parts = command.split()
                    if len(parts) >= 2:
                        try:
                            cluster_id = int(parts[1])
                            limit = int(parts[2]) if len(parts) > 2 else 200
                            self.plot_2d_cluster(cluster_id, limit)
                        except ValueError:
                            print("Invalid cluster ID or limit")
                    else:
                        print("Usage: plot2d <cluster_id> [limit]")
                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def show_help(self):
        """Show help information."""
        print("\n=== Available Commands ===")
        print("list              - List all clusters with statistics")
        print("stats             - Show detailed projection progress")
        print("tree              - List cluster tree information")
        print("radial            - Create radial dendrogram visualization")
        print("traditional       - Create traditional dendrogram visualization")
        print("analyze           - Analyze cluster tree data quality and issues")
        print("validate          - Validate cluster tree integrity")
        print("plot <id> [limit] - Create 3D visualization of cluster (default limit: 100)")
        print("plot2d <id> [limit] - Create 2D visualization of cluster (default limit: 200)")
        print("help              - Show this help message")
        print("quit/exit/q       - Exit interactive mode")
        print("\nExamples:")
        print("  plot 5")
        print("  plot 25 50")
        print("  plot2d 10")
        print("  radial")
        print("  traditional")
        print("  analyze")
        print("  validate")
        print("  radial --max-depth 3")
        print("  traditional --max-depth 3 --max-nodes 50")
        print("  tree")

    def get_cluster_tree_nodes(self, namespace: str = 'enwiki_namespace_0') -> List[Dict]:
        """Get all cluster tree nodes for a specific namespace."""
        conn = self.get_connection()

        query = """
            SELECT
                node_id,
                namespace,
                parent_id,
                depth,
                doc_count,
                child_count,
                first_label,
                final_label
            FROM cluster_tree
            WHERE namespace = ?
            ORDER BY depth ASC, node_id ASC
        """

        try:
            cursor = conn.execute(query, (namespace,))
            rows = cursor.fetchall()
            nodes = []
            for row in rows:
                nodes.append({
                    'node_id': row['node_id'],
                    'namespace': row['namespace'],
                    'parent_id': row['parent_id'],
                    'depth': row['depth'],
                    'doc_count': row['doc_count'],
                    'child_count': row['child_count'],
                    'first_label': row['first_label'],
                    'final_label': row['final_label']
                })
            logger.debug(f"Found {len(nodes)} cluster tree nodes for namespace {namespace}")
            return nodes
        except Exception as e:
            logger.error(f"Error getting cluster tree nodes: {e}")
            raise

    def calculate_descendant_counts(self, nodes: List[Dict]) -> Dict[int, int]:
        """Calculate the total number of descendant children for each node."""
        # Build parent-child relationships
        children_map = defaultdict(list)
        for node in nodes:
            if node['parent_id'] is not None:
                children_map[node['parent_id']].append(node['node_id'])

        # Calculate descendant counts using DFS
        descendant_counts = {}

        def dfs(node_id):
            total_descendants = 0
            for child_id in children_map.get(node_id, []):
                total_descendants += 1 + dfs(child_id)
            descendant_counts[node_id] = total_descendants
            return total_descendants

        # Start from root nodes (nodes with no parent)
        root_nodes = [node for node in nodes if node['parent_id'] is None]
        for root in root_nodes:
            dfs(root['node_id'])

        return descendant_counts

    def plot_radial_dendrogram(self, namespace: str = 'enwiki_namespace_0',
                             max_depth: Optional[int] = None,
                             max_nodes: int = 100,
                             save_path: Optional[str] = None):
        """Create a radial dendrogram visualization of the cluster tree."""
        # Get cluster tree nodes
        nodes = self.get_cluster_tree_nodes(namespace)

        if not nodes:
            print(f"No cluster tree nodes found for namespace {namespace}")
            return

        # Filter by max depth if specified
        if max_depth is not None:
            nodes = [node for node in nodes if node['depth'] <= max_depth]

        # Limit the number of nodes for better visualization
        if len(nodes) > max_nodes:
            print(f"Warning: Too many nodes ({len(nodes)}). Limiting to {max_nodes} nodes for visualization.")
            # Sort by doc_count and take the top nodes
            nodes = sorted(nodes, key=lambda x: x['doc_count'], reverse=True)[:max_nodes]

        # Calculate descendant counts
        descendant_counts = self.calculate_descendant_counts(nodes)

        # Build tree structure with improved approach
        children_map = defaultdict(list)
        parent_map = {}
        node_id_to_node = {node['node_id']: node for node in nodes}

        for node in nodes:
            if node['parent_id'] is not None and node['parent_id'] in node_id_to_node:
                children_map[node['parent_id']].append(node)
                parent_map[node['node_id']] = node['parent_id']

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        # Colors for different depths
        cmap = plt.get_cmap('viridis')
        max_depth_val = max([node['depth'] for node in nodes]) + 1
        colors = [cmap(i / max(1, max_depth_val - 1)) for i in range(max_depth_val)]

        # Sort nodes by depth for proper hierarchical ordering
        sorted_nodes = sorted(nodes, key=lambda x: x['depth'])

        # Plot each level using improved method
        for root in [node for node in nodes if node['parent_id'] is None]:
            self._plot_radial_node(ax, root, children_map, descendant_counts, colors, 0)

        # Customize the plot
        ax.set_title(f'Radial Dendrogram - {namespace}\n(Labeled by Descendant Children Count)',
                    fontsize=14, pad=20)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_axis_off()

        # Add legend
        depth_levels = sorted(set(node['depth'] for node in nodes))
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=colors[depth], markersize=10,
                                     label=f'Depth {depth}')
                           for depth in depth_levels]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Radial dendrogram saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_traditional_dendrogram(self, namespace: str = 'enwiki_namespace_0',
                                   max_depth: Optional[int] = None,
                                   max_nodes: int = 100,
                                   save_path: Optional[str] = None):
        """Create a traditional dendrogram visualization of the cluster tree."""
        # Get cluster tree nodes
        nodes = self.get_cluster_tree_nodes(namespace)

        if not nodes:
            print(f"No cluster tree nodes found for namespace {namespace}")
            return

        # Filter by max depth if specified
        if max_depth is not None:
            nodes = [node for node in nodes if node['depth'] <= max_depth]

        # Limit the number of nodes for better visualization
        if len(nodes) > max_nodes:
            print(f"Warning: Too many nodes ({len(nodes)}). Limiting to {max_nodes} nodes for visualization.")
            # Sort by doc_count and take the top nodes
            nodes = sorted(nodes, key=lambda x: x['doc_count'], reverse=True)[:max_nodes]

        # Build tree structure
        children_map = defaultdict(list)
        parent_map = {}
        node_id_to_node = {node['node_id']: node for node in nodes}

        for node in nodes:
            if node['parent_id'] is not None and node['parent_id'] in node_id_to_node:
                children_map[node['parent_id']].append(node['node_id'])
                parent_map[node['node_id']] = node['parent_id']

        if not parent_map:
            print("No parent-child relationships found for dendrogram")
            return

        # Create a proper linkage matrix using a simpler approach
        # We'll use scipy's linkage function with a distance matrix

        # Create a mapping from node_id to index (0 to n-1)
        node_ids = [node['node_id'] for node in nodes]
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        n_nodes = len(nodes)

        # Create distance matrix
        distance_matrix = np.zeros((n_nodes, n_nodes))

        # Set distances based on tree relationships
        for i, node in enumerate(nodes):
            if node['node_id'] in parent_map:
                parent_id = parent_map[node['node_id']]
                if parent_id in node_id_to_index:
                    j = node_id_to_index[parent_id]
                    # Distance based on depth difference
                    depth_diff = node['depth'] - nodes[j]['depth']
                    distance_matrix[i, j] = depth_diff * 10  # Weight depth heavily
                    distance_matrix[j, i] = depth_diff * 10

        # Fill in remaining distances with large values
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and distance_matrix[i, j] == 0:
                    # Use depth difference as fallback
                    depth_diff = abs(nodes[i]['depth'] - nodes[j]['depth'])
                    distance_matrix[i, j] = depth_diff * 10 + 100  # Add base distance

        # Convert to condensed distance matrix (upper triangle)
        condensed_dist = pdist(distance_matrix)

        # Use scipy's linkage function to create proper linkage matrix
        try:
            linkage_matrix = linkage(condensed_dist, method='ward')
        except Exception as e:
            print(f"Error creating linkage matrix: {e}")
            # Fallback to single linkage
            linkage_matrix = linkage(condensed_dist, method='single')

        # Create the dendrogram
        plt.figure(figsize=(15, 10))

        # Create labels for the dendrogram
        labels = []
        for i, node in enumerate(nodes):
            label = f"Node {node['node_id']}"
            if node['first_label']:
                label = f"{node['first_label']}"
            if node['doc_count'] > 0:
                label += f" ({node['doc_count']})"
            labels.append(label)

        # Plot dendrogram
        dendrogram(linkage_matrix,
                  labels=labels,
                  orientation='top',
                  show_leaf_counts=True,
                   leaf_rotation=90,
                   leaf_font_size=8,
                   show_contracted=False)

        plt.title(f'Traditional Dendrogram - {namespace}\n(Cluster Hierarchy by Depth Relationships)',
                 fontsize=14, pad=20)
        plt.xlabel('Cluster Nodes', fontsize=12)
        plt.ylabel('Distance (based on depth relationships)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Traditional dendrogram saved to {save_path}")

        plt.show()

    def plot_hierarchical_cluster(self, namespace: str = 'enwiki_namespace_0',
                                max_depth: Optional[int] = None,
                                max_nodes: int = 100,
                                save_path: Optional[str] = None):
        """Create a hierarchical cluster visualization using better layout algorithms."""
        # Get cluster tree nodes
        nodes = self.get_cluster_tree_nodes(namespace)

        if not nodes:
            print(f"No cluster tree nodes found for namespace {namespace}")
            return

        # Filter by max depth if specified
        if max_depth is not None:
            nodes = [node for node in nodes if node['depth'] <= max_depth]

        # Limit the number of nodes for better visualization
        if len(nodes) > max_nodes:
            print(f"Warning: Too many nodes ({len(nodes)}). Limiting to {max_nodes} nodes for visualization.")
            nodes = sorted(nodes, key=lambda x: x['doc_count'], reverse=True)[:max_nodes]

        # Build tree structure
        children_map = defaultdict(list)
        parent_map = {}
        node_id_to_node = {node['node_id']: node for node in nodes}

        for node in nodes:
            if node['parent_id'] is not None and node['parent_id'] in node_id_to_node:
                children_map[node['parent_id']].append(node['node_id'])
                parent_map[node['node_id']] = node['parent_id']

        # Create figure with subplots for different visualization approaches
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Cluster Tree Visualization - {namespace}', fontsize=16, fontweight='bold')

        # 1. Traditional Top-Down Dendrogram
        ax1 = axes[0, 0]
        self._plot_top_down_dendrogram(ax1, nodes, parent_map, children_map)
        ax1.set_title('Traditional Top-Down Dendrogram')

        # 2. Level-by-Level Visualization
        ax2 = axes[0, 1]
        self._plot_level_by_level(ax2, nodes, children_map)
        ax2.set_title('Level-by-Level Visualization')

        # 3. Rectangular Tree Layout
        ax3 = axes[1, 0]
        self._plot_rectangular_tree(ax3, nodes, children_map)
        ax3.set_title('Rectangular Tree Layout')

        # 4. Radial (but improved with fewer nodes)
        ax4 = axes[1, 1]
        if len(nodes) <= 25:  # Only use radial for small trees
            self._plot_improved_radial(ax4, nodes, children_map)
            ax4.set_title('Improved Radial Layout (Small Trees Only)')
        else:
            ax4.text(0.5, 0.5, 'Radial layout disabled\nfor large trees',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Radial Layout (Disabled)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hierarchical cluster visualization saved to {save_path}")

        plt.show()

    def _plot_top_down_dendrogram(self, ax, nodes, parent_map, children_map):
        """Plot a traditional top-down dendrogram."""
        # Sort nodes by depth
        sorted_nodes = sorted(nodes, key=lambda x: x['depth'])

        # Create simple hierarchical layout
        y_pos = {}
        current_y = 0

        for depth in range(max(node['depth'] for node in nodes) + 1):
            depth_nodes = [node for node in nodes if node['depth'] == depth]
            for i, node in enumerate(depth_nodes):
                y_pos[node['node_id']] = current_y + i
            current_y += len(depth_nodes) + 1

        # Draw connections
        for node in nodes:
            if node['node_id'] in parent_map:
                parent_id = parent_map[node['node_id']]
                parent_node = next(n for n in nodes if n['node_id'] == parent_id)
                ax.plot([parent_node['depth'], node['depth']],
                       [y_pos[parent_id], y_pos[node['node_id']]], 'b-', alpha=0.6)

        # Draw nodes
        for node in sorted_nodes:
            ax.scatter(node['depth'], y_pos[node['node_id']],
                      c=node['depth'], cmap='viridis', s=50, alpha=0.8)
            ax.annotate(f"N{node['node_id']}",
                       (node['depth'], y_pos[node['node_id']]),
                       xytext=(5, 0), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Depth')
        ax.set_ylabel('Node Position')
        ax.grid(True, alpha=0.3)

    def _plot_level_by_level(self, ax, nodes, children_map):
        """Plot nodes level by level with clear separation."""
        max_depth = max(node['depth'] for node in nodes)

        for depth in range(max_depth + 1):
            depth_nodes = [node for node in nodes if node['depth'] == depth]
            y_positions = np.linspace(0, 10, len(depth_nodes) + 2)[1:-1]

            for i, node in enumerate(depth_nodes):
                x = depth
                y = y_positions[i]

                # Draw node
                ax.scatter(x, y, c=depth, cmap='viridis', s=100, alpha=0.8, edgecolors='black')

                # Draw connections to children
                for child_id in children_map.get(node['node_id'], []):
                    if child_id in [n['node_id'] for n in nodes]:
                        child_node = next(n for n in nodes if n['node_id'] == child_id)
                        child_y = np.linspace(0, 10, len([n for n in nodes if n['depth'] == child_node['depth']]) + 2)[1:-1]
                        child_idx = [n['node_id'] for n in nodes if n['depth'] == child_node['depth']].index(child_id)
                        ax.plot([x, child_node['depth']], [y, child_y[child_idx]], 'b-', alpha=0.6)

                # Label
                ax.annotate(f"N{node['node_id']}", (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, ha='left')

        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Node Position')
        ax.grid(True, alpha=0.3)

    def _plot_rectangular_tree(self, ax, nodes, children_map):
        """Plot tree using rectangular layout (better for large trees)."""
        # Calculate positions using a tree layout algorithm
        node_positions = {}

        def calculate_positions(node, x=0.0, y=0.0, level_width=1.0):
            node_positions[node['node_id']] = (x, y)

            children = children_map.get(node['node_id'], [])
            if children:
                child_width = level_width / len(children)
                for i, child_id in enumerate(children):
                    try:
                        child_node = next(n for n in nodes if n['node_id'] == child_id)
                        new_x = x + (i - len(children)/2 + 0.5) * child_width
                        calculate_positions(child_node, new_x, y - 1, child_width)
                    except StopIteration:
                        # Skip child nodes that are not in the current filtered set
                        continue

        # Start from root nodes
        root_nodes = [node for node in nodes if node['parent_id'] is None]
        for i, root in enumerate(root_nodes):
            calculate_positions(root, i * 2, 0, 2)

        # Draw connections
        for node in nodes:
            if node['node_id'] in node_positions:
                x, y = node_positions[node['node_id']]
                for child_id in children_map.get(node['node_id'], []):
                    if child_id in node_positions:
                        child_x, child_y = node_positions[child_id]
                        ax.plot([x, child_x], [y, child_y], 'b-', alpha=0.6, linewidth=1)

        # Draw nodes
        for node in nodes:
            if node['node_id'] in node_positions:
                x, y = node_positions[node['node_id']]
                ax.scatter(x, y, c=node['depth'], cmap='viridis', s=80, alpha=0.8, edgecolors='black')
                ax.annotate(f"N{node['node_id']}", (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, ha='left')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position (Depth)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _plot_improved_radial(self, ax, nodes, children_map):
        """Plot radial layout but with better spacing for small trees."""
        # Calculate positions
        node_positions = {}

        def calculate_radial_positions(node, angle=0.0, radius=0.0, angle_span=np.pi/3):
            node_positions[node['node_id']] = (radius, angle)

            children = children_map.get(node['node_id'], [])
            if children:
                child_angle_span = angle_span / len(children)
                for i, child_id in enumerate(children):
                    child_node = next(n for n in nodes if n['node_id'] == child_id)
                    child_angle = angle + (i - len(children)/2 + 0.5) * child_angle_span
                    calculate_radial_positions(child_node, child_angle, radius + 1, child_angle_span)

        # Start from root nodes
        root_nodes = [node for node in nodes if node['parent_id'] is None]
        for i, root in enumerate(root_nodes):
            angle = i * 2 * np.pi / len(root_nodes)
            calculate_radial_positions(root, angle, 0, 2 * np.pi / len(root_nodes))

        # Convert to cartesian and draw
        for node in nodes:
            if node['node_id'] in node_positions:
                radius, angle = node_positions[node['node_id']]
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                # Draw connections
                for child_id in children_map.get(node['node_id'], []):
                    if child_id in node_positions:
                        child_radius, child_angle = node_positions[child_id]
                        child_x = child_radius * np.cos(child_angle)
                        child_y = child_radius * np.sin(child_angle)
                        ax.plot([x, child_x], [y, child_y], 'b-', alpha=0.6, linewidth=1)

                # Draw nodes
                ax.scatter(x, y, c=node['depth'], cmap='viridis', s=80, alpha=0.8, edgecolors='black')
                ax.annotate(f"N{node['node_id']}", (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, ha='left')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def _plot_radial_node(self, ax, node, children_map, descendant_counts, colors, angle_offset):
        """Recursively plot a single node and its children in radial layout."""
        node_id = node['node_id']
        depth = node['depth']
        doc_count = node['doc_count']
        descendant_count = descendant_counts.get(node_id, 0)

        # Calculate radius based on depth
        radius = 0.1 + (depth * 0.15)

        # Calculate angle span for this node
        children = children_map.get(node_id, [])
        if children:
            angle_span = 2 * np.pi / len(children)
        else:
            angle_span = 2 * np.pi

        # Calculate starting angle
        if node['parent_id'] is None:
            # Root node starts at top
            start_angle = -np.pi/2
        else:
            # For child nodes, we need to calculate based on parent position
            # This is simplified - in a full implementation, you'd track parent angles
            start_angle = angle_offset

        # Draw the node circle
        circle_angles = np.linspace(start_angle, start_angle + angle_span, 100)
        circle_x = radius * np.cos(circle_angles)
        circle_y = radius * np.sin(circle_angles)
        ax.fill(circle_x, circle_y, color=colors[depth], alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add label with descendant count
        label_angle = start_angle + angle_span / 2
        label_radius = radius + 0.05
        ax.text(label_radius * np.cos(label_angle),
                label_radius * np.sin(label_angle),
                f"{descendant_count}",
                ha='center', va='center', fontsize=8, fontweight='bold')

        # Draw connecting line to parent (for non-root nodes)
        if node['parent_id'] is not None:
            parent_radius = 0.1 + ((depth - 1) * 0.15)
            ax.plot([0, radius * np.cos(label_angle)],
                   [0, radius * np.sin(label_angle)],
                   'k-', alpha=0.3, linewidth=0.5)

        # Recursively plot children
        if children:
            child_angle_offset = start_angle
            for child in children:
                child_angle_span = angle_span / len(children)
                self._plot_radial_node(ax, child, children_map, descendant_counts,
                                      colors, child_angle_offset)
                child_angle_offset += child_angle_span

    def list_cluster_tree_info(self, namespace: str = 'enwiki_namespace_0'):
        """List information about the cluster tree structure."""
        nodes = self.get_cluster_tree_nodes(namespace)

        if not nodes:
            print(f"No cluster tree nodes found for namespace {namespace}")
            return

        # Calculate statistics
        depths = [node['depth'] for node in nodes]
        total_nodes = len(nodes)
        max_depth = max(depths)
        root_nodes = [node for node in nodes if node['parent_id'] is None]
        leaf_nodes = [node for node in nodes if not node['child_count'] or node['child_count'] == 0]

        print(f"\n=== Cluster Tree Information for {namespace} ===")
        print(f"Total nodes: {total_nodes}")
        print(f"Root nodes: {len(root_nodes)}")
        print(f"Leaf nodes: {len(leaf_nodes)}")
        print(f"Maximum depth: {max_depth}")
        print(f"Depth distribution:")

        for depth in range(max_depth + 1):
            count = len([node for node in nodes if node['depth'] == depth])
            print(f"  Depth {depth}: {count} nodes")

        # Show root nodes
        print(f"\nRoot nodes:")
        for root in root_nodes:
            descendant_count = self.calculate_descendant_counts([root]).get(root['node_id'], 0)
            print(f"  Node {root['node_id']}: {root['doc_count']} documents, {descendant_count} descendants")

    def analyze_cluster_tree_quality(self, namespace: str = 'enwiki_namespace_0'):
        """Analyze the quality and potential issues with cluster tree data."""
        print(f"\n=== Cluster Tree Quality Analysis for {namespace} ===")

        nodes = self.get_cluster_tree_nodes(namespace)
        if not nodes:
            print("No cluster tree nodes found")
            return

        # Build parent-child relationships
        children_map = defaultdict(list)
        parent_map = {}
        for node in nodes:
            if node['parent_id'] is not None:
                children_map[node['parent_id']].append(node['node_id'])
                parent_map[node['node_id']] = node['parent_id']

        # 1. Check for cycles
        print("\n1. Cycle Detection:")
        visited = set()
        rec_stack = set()
        has_cycles = False

        def dfs_cycle(node_id):
            nonlocal has_cycles
            visited.add(node_id)
            rec_stack.add(node_id)

            if node_id in children_map:
                for child_id in children_map[node_id]:
                    if child_id not in visited:
                        if dfs_cycle(child_id):
                            return True
                    elif child_id in rec_stack:
                        print(f"   Cycle detected: {child_id} -> ... -> {node_id}")
                        has_cycles = True
                        return True

            rec_stack.remove(node_id)
            return False

        for node in nodes:
            if node['node_id'] not in visited:
                if dfs_cycle(node['node_id']):
                    break

        if not has_cycles:
            print("   No cycles detected")

        # 2. Check for disconnected components
        print("\n2. Connectivity Analysis:")
        visited = set()
        components = 0

        def dfs_component(node_id):
            stack = [node_id]
            component = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    # Add children
                    if current in children_map:
                        for child in children_map[current]:
                            if child not in visited:
                                stack.append(child)
                    # Add parent (if not root)
                    if current in parent_map:
                        parent = parent_map[current]
                        if parent not in visited:
                            stack.append(parent)
            return component

        components = []
        for node in nodes:
            if node['node_id'] not in visited:
                component = dfs_component(node['node_id'])
                components.append(component)

        print(f"   Number of connected components: {len(components)}")
        for i, component in enumerate(components):
            root_nodes_in_component = [n for n in component if n not in parent_map]
            print(f"   Component {i+1}: {len(component)} nodes, {len(root_nodes_in_component)} root nodes")

        # 3. Check depth consistency
        print("\n3. Depth Consistency Analysis:")
        depth_inconsistencies = []

        for node in nodes:
            if node['parent_id'] is not None and node['parent_id'] in parent_map:
                parent_depth = None
                for parent_node in nodes:
                    if parent_node['node_id'] == node['parent_id']:
                        parent_depth = parent_node['depth']
                        break

                if parent_depth is not None and node['depth'] <= parent_depth:
                    depth_inconsistencies.append((node['node_id'], node['parent_id'], parent_depth, node['depth']))

        if depth_inconsistencies:
            print(f"   Found {len(depth_inconsistencies)} depth inconsistencies:")
            for child_id, parent_id, parent_depth, child_depth in depth_inconsistencies[:10]:  # Show first 10
                print(f"     Node {child_id} (depth {child_depth}) has parent {parent_id} (depth {parent_depth})")
        else:
            print("   No depth inconsistencies found")

        # 4. Check document count consistency
        print("\n4. Document Count Analysis:")
        doc_count_issues = []

        for node in nodes:
            # Check if doc_count is reasonable
            if node['doc_count'] < 0:
                doc_count_issues.append((node['node_id'], node['doc_count'], "Negative document count"))

            # Check if child doc counts sum to parent (approximately)
            if node['node_id'] in children_map:
                children_doc_count = sum(n['doc_count'] for n in nodes if n['node_id'] in children_map[node['node_id']])
                if children_doc_count > 0 and node['doc_count'] > 0:
                    ratio = children_doc_count / node['doc_count']
                    if ratio > 1.1:  # Children have significantly more documents than parent
                        doc_count_issues.append((node['node_id'], node['doc_count'],
                                              f"Children have {children_doc_count} docs vs parent {node['doc_count']} (ratio: {ratio:.2f})"))

        if doc_count_issues:
            print(f"   Found {len(doc_count_issues)} document count issues:")
            for node_id, doc_count, issue in doc_count_issues[:10]:  # Show first 10
                print(f"     Node {node_id}: {issue}")
        else:
            print("   No obvious document count issues found")

        # 5. Check tree balance
        print("\n5. Tree Balance Analysis:")
        depths = [node['depth'] for node in nodes]
        if depths:
            max_depth = max(depths)
            min_depth = min(depths)
            avg_depth = sum(depths) / len(depths)

            print(f"   Depth range: {min_depth} to {max_depth}")
            print(f"   Average depth: {avg_depth:.2f}")
            print(f"   Depth standard deviation: {np.std(depths):.2f}")

            # Check for very deep trees
            deep_nodes = [n for n in nodes if n['depth'] > max_depth * 0.8]
            if deep_nodes:
                print(f"   Warning: {len(deep_nodes)} nodes are in the deepest 20% of the tree")

        # 6. Check for potential data quality issues
        print("\n6. Data Quality Checks:")
        issues = []

        # Check for null/missing values
        for i, node in enumerate(nodes):
            if node['first_label'] is None and node['final_label'] is None:
                issues.append((node['node_id'], "Both first_label and final_label are null"))
            if node['doc_count'] is None:
                issues.append((node['node_id'], "doc_count is null"))

        if issues:
            print(f"   Found {len(issues)} potential data quality issues:")
            for node_id, issue in issues[:10]:  # Show first 10
                print(f"     Node {node_id}: {issue}")
        else:
            print("   No obvious data quality issues found")

        # 7. Summary statistics
        print("\n7. Summary Statistics:")
        total_docs = sum(n['doc_count'] for n in nodes if n['doc_count'] is not None)
        total_nodes = len(nodes)
        avg_docs_per_node = total_docs / total_nodes if total_nodes > 0 else 0

        print(f"   Total nodes: {total_nodes}")
        print(f"   Total documents: {total_docs}")
        print(f"   Average documents per node: {avg_docs_per_node:.2f}")

        # Calculate tree density
        total_edges = len(parent_map)
        max_possible_edges = total_nodes * (total_nodes - 1) // 2
        tree_density = total_edges / max_possible_edges if max_possible_edges > 0 else 0

        print(f"   Tree edges: {total_edges}")
        print(f"   Tree density: {tree_density:.4f} (should be close to 0 for a tree)")

    def validate_cluster_tree_integrity(self, namespace: str = 'enwiki_namespace_0'):
        """Validate the integrity of the cluster tree structure."""
        print(f"\n=== Cluster Tree Integrity Validation for {namespace} ===")

        nodes = self.get_cluster_tree_nodes(namespace)
        if not nodes:
            print("No cluster tree nodes found")
            return False

        issues_found = []

        # Build relationships
        children_map = defaultdict(list)
        parent_map = {}
        node_id_to_node = {node['node_id']: node for node in nodes}

        for node in nodes:
            if node['parent_id'] is not None:
                children_map[node['parent_id']].append(node['node_id'])
                parent_map[node['node_id']] = node['parent_id']

        # Check 1: All parent references exist
        print("\n1. Parent Reference Validation:")
        invalid_parents = []
        for node_id, parent_id in parent_map.items():
            if parent_id not in node_id_to_node:
                invalid_parents.append((node_id, parent_id))

        if invalid_parents:
            print(f"   Found {len(invalid_parents)} nodes with invalid parent references:")
            for node_id, parent_id in invalid_parents[:10]:
                print(f"     Node {node_id} references non-existent parent {parent_id}")
            issues_found.append("Invalid parent references")
        else:
            print("   All parent references are valid")

        # Check 2: Root nodes have no parents
        print("\n2. Root Node Validation:")
        root_nodes = [node for node in nodes if node['parent_id'] is None]
        print(f"   Found {len(root_nodes)} root nodes")

        if len(root_nodes) == 0:
            print("   WARNING: No root nodes found - this indicates a potential cycle or all nodes have parents")
            issues_found.append("No root nodes")
        elif len(root_nodes) > 1:
            print(f"   INFO: Multiple root nodes found ({len(root_nodes)}) - this is normal for a forest")

        # Check 3: Depth consistency
        print("\n3. Depth Consistency Check:")
        depth_violations = []
        for node in nodes:
            if node['parent_id'] is not None and node['parent_id'] in node_id_to_node:
                parent = node_id_to_node[node['parent_id']]
                if node['depth'] <= parent['depth']:
                    depth_violations.append((node['node_id'], node['depth'], parent['node_id'], parent['depth']))

        if depth_violations:
            print(f"   Found {len(depth_violations)} depth violations:")
            for child_id, child_depth, parent_id, parent_depth in depth_violations[:10]:
                print(f"     Node {child_id} (depth {child_depth}) should have depth > parent {parent_id} (depth {parent_depth})")
            issues_found.append("Depth violations")
        else:
            print("   All depths are consistent with parent-child relationships")

        # Check 4: Document count inheritance
        print("\n4. Document Count Inheritance Check:")
        doc_count_issues = []

        for node in nodes:
            if node['node_id'] in children_map:
                children_total = sum(node_id_to_node[child_id]['doc_count']
                                   for child_id in children_map[node['node_id']]
                                   if child_id in node_id_to_node)

                if node['doc_count'] is not None and children_total > 0:
                    if children_total > node['doc_count'] * 1.5:  # Children significantly exceed parent
                        doc_count_issues.append((node['node_id'], node['doc_count'], children_total))

        if doc_count_issues:
            print(f"   Found {len(doc_count_issues)} nodes where children document count significantly exceeds parent:")
            for node_id, parent_count, children_count in doc_count_issues[:10]:
                print(f"     Node {node_id}: parent has {parent_count}, children have {children_count}")
            issues_found.append("Document count inheritance issues")
        else:
            print("   Document count inheritance appears reasonable")

        # Check 5: Tree structure validation
        print("\n5. Tree Structure Validation:")
        visited = set()
        tree_nodes = 0

        def validate_tree_structure(node_id):
            nonlocal tree_nodes
            visited.add(node_id)
            tree_nodes += 1

            if node_id in children_map:
                for child_id in children_map[node_id]:
                    if child_id not in visited:
                        validate_tree_structure(child_id)

        # Validate from each root
        for root in root_nodes:
            validate_tree_structure(root['node_id'])

        if tree_nodes != len(nodes):
            print(f"   WARNING: Tree validation covered {tree_nodes} nodes, but total nodes are {len(nodes)}")
            print("   This may indicate disconnected components or cycles")
            issues_found.append("Disconnected components")
        else:
            print("   Tree structure is valid and connected")

        # Final summary
        print(f"\n=== Integrity Validation Summary ===")
        if issues_found:
            print(f"Found {len(issues_found)} types of issues:")
            for issue in issues_found:
                print(f"  - {issue}")
            print("\nRecommendations:")
            print("  1. Check data loading and cluster tree generation process")
            print("  2. Verify the clustering algorithm produces valid tree structures")
            print("  3. Consider data cleaning or validation steps before clustering")
            return False
        else:
            print("All integrity checks passed!")
            return True


def main():
    parser = argparse.ArgumentParser(description='Visualize Wikipedia embeddings')
    parser.add_argument('--cluster', type=int, help='Cluster ID to visualize')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of pages to display')
    parser.add_argument('--two-d', action='store_true', help='Create 2D visualization instead of 3D')
    parser.add_argument('--list-clusters', action='store_true', help='List all clusters')
    parser.add_argument('--stats', action='store_true', help='Show cluster statistics')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--radial-dendrogram', action='store_true', help='Create radial dendrogram visualization')
    parser.add_argument('--traditional-dendrogram', action='store_true', help='Create traditional dendrogram visualization')
    parser.add_argument('--hierarchical', action='store_true', help='Create hierarchical cluster visualization with multiple layouts')
    parser.add_argument('--analyze-tree', action='store_true', help='Analyze cluster tree data quality and issues')
    parser.add_argument('--validate-tree', action='store_true', help='Validate cluster tree integrity')
    parser.add_argument('--namespace', default='enwiki_namespace_0', help='Namespace to filter by')
    parser.add_argument('--max-depth', type=int, help='Maximum depth to show in dendrogram')
    parser.add_argument('--max-nodes', type=int, default=100, help='Maximum number of nodes to show in dendrogram')
    parser.add_argument('--save-path', help='Path to save the visualization')
    parser.add_argument('--list-tree', action='store_true', help='List cluster tree information')
    parser.add_argument('--db', default='chunk_log.db', help='Path to SQLite database')

    args = parser.parse_args()

    visualizer = EmbeddingVisualizer(args.db)

    try:
        logger.debug(f"Arguments received: cluster={args.cluster}, 2d={args.two_d}, limit={args.limit}, "
                    f"radial_dendrogram={args.radial_dendrogram}, traditional_dendrogram={args.traditional_dendrogram}, "
                    f"analyze_tree={args.analyze_tree}, validate_tree={args.validate_tree}, namespace={args.namespace}")

        if args.list_clusters:
            logger.debug("Executing list_clusters command")
            visualizer.list_clusters()
        elif args.stats:
            logger.debug("Executing show_cluster_statistics command")
            visualizer.show_cluster_statistics()
        elif args.list_tree:
            logger.debug("Executing list_cluster_tree_info command")
            visualizer.list_cluster_tree_info(args.namespace)
        elif args.analyze_tree:
            logger.debug(f"Executing cluster tree quality analysis for namespace {args.namespace}")
            visualizer.analyze_cluster_tree_quality(args.namespace)
        elif args.validate_tree:
            logger.debug(f"Executing cluster tree integrity validation for namespace {args.namespace}")
            visualizer.validate_cluster_tree_integrity(args.namespace)
        elif args.radial_dendrogram:
            logger.debug(f"Executing radial dendrogram visualization for namespace {args.namespace}")
            visualizer.plot_radial_dendrogram(args.namespace, args.max_depth, args.max_nodes, args.save_path)
        elif args.traditional_dendrogram:
            logger.debug(f"Executing traditional dendrogram visualization for namespace {args.namespace}")
            visualizer.plot_traditional_dendrogram(args.namespace, args.max_depth, args.max_nodes, args.save_path)
        elif args.hierarchical:
            logger.debug(f"Executing hierarchical cluster visualization for namespace {args.namespace}")
            visualizer.plot_hierarchical_cluster(args.namespace, args.max_depth, args.max_nodes, args.save_path)
        elif args.cluster is not None:
            logger.debug(f"Executing visualization for cluster {args.cluster}, 2d={args.two_d}")
            if args.two_d:
                visualizer.plot_2d_cluster(args.cluster, args.limit)
            else:
                visualizer.plot_3d_cluster(args.cluster, args.limit)
        elif args.interactive:
            logger.debug("Starting interactive mode")
            visualizer.interactive_mode()
        else:
            logger.debug("No action specified, showing help and cluster list")
            print("No action specified. Use --help for usage information.")
            visualizer.list_clusters()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        logger.debug("Closing database connection")
        visualizer.close()


if __name__ == "__main__":
    main()
