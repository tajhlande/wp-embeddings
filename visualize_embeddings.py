#!/usr/bin/env python3
"""
Wikipedia Embeddings Visualization Tool

This script provides command-line and interactive visualization capabilities for
the 3D embeddings stored in the SQLite database.

Usage:
    python visualize_embeddings.py --cluster <cluster_id> [--limit <n>] [--2d]
    python visualize_embeddings.py --list-clusters
    python visualize_embeddings.py --interactive
"""

import argparse
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional
import logging

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
        print("plot <id> [limit] - Create 3D visualization of cluster (default limit: 100)")
        print("plot2d <id> [limit] - Create 2D visualization of cluster (default limit: 200)")
        print("help              - Show this help message")
        print("quit/exit/q       - Exit interactive mode")
        print("\nExamples:")
        print("  plot 5")
        print("  plot 25 50")
        print("  plot2d 10")


def main():
    parser = argparse.ArgumentParser(description='Visualize Wikipedia embeddings')
    parser.add_argument('--cluster', type=int, help='Cluster ID to visualize')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of pages to display')
    parser.add_argument('--two-d', action='store_true', help='Create 2D visualization instead of 3D')
    parser.add_argument('--list-clusters', action='store_true', help='List all clusters')
    parser.add_argument('--stats', action='store_true', help='Show cluster statistics')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--db', default='chunk_log.db', help='Path to SQLite database')

    args = parser.parse_args()

    visualizer = EmbeddingVisualizer(args.db)

    try:
        logger.debug(f"Arguments received: cluster={args.cluster}, 2d={args.two_d}, limit={args.limit}")

        if args.list_clusters:
            logger.debug("Executing list_clusters command")
            visualizer.list_clusters()
        elif args.stats:
            logger.debug("Executing show_cluster_statistics command")
            visualizer.show_cluster_statistics()
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