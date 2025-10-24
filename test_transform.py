import sqlite3
import numpy as np
from transform import run_pca, run_kmeans
from database import ensure_tables

def test_pca_batch_sufficient():
    # Create a test database with proper schema
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Ensure tables are created with proper schema
    ensure_tables(conn)

    # Insert some test data - create 150 vectors of 2048 dimensions each
    # This should be enough to satisfy the n_components=100 requirement
    namespace = 'my_namespace'
    chunk_name = 'my_chunk'
    conn.execute('INSERT INTO chunk_log (chunk_name, namespace) VALUES (?, ?)', 
                (chunk_name, namespace))
    for i in range(150):
        vector = np.random.randn(2048).astype(np.float32)
        conn.execute('INSERT INTO page_log (page_id, chunk_name) VALUES (?, ?)',
                     (i, chunk_name))
        conn.execute('INSERT INTO page_vector (page_id, embedding_vector) VALUES (?, ?)', 
                    (i, vector.tobytes()))

    conn.commit()

    # Test the run_pca function
    try:
        batch_count, total_vectors = run_pca(conn, namespace, target_dim=100, batch_size=50)
        print(f'Success! Processed {total_vectors} vectors in {batch_count} batches')
        
        # Verify that reduced vectors were stored
        cursor = conn.execute('SELECT COUNT(*) as count FROM page_vector WHERE reduced_vector IS NOT NULL')
        result = cursor.fetchone()
        print(f'Reduced vectors stored: {result["count"]}')
        
    # except Exception as e:
    #     print(f'Error: {e}')
    #     import traceback
    #     traceback.print_exc()
    finally:
        conn.close()
    
def test_pca_insufficient_batch():
    # Create a test database with proper schema
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Ensure tables are created with proper schema
    ensure_tables(conn)

    namespace = 'my_namespace'
    chunk_name = 'my_chunk'
    conn.execute('INSERT INTO chunk_log (chunk_name, namespace) VALUES (?, ?)', 
                (chunk_name, namespace))

    # Insert some test data - create 150 vectors of 2048 dimensions each
    for i in range(150):
        vector = np.random.randn(2048).astype(np.float32)
        conn.execute('INSERT INTO page_log (page_id, chunk_name) VALUES (?, ?)',
                     (i, chunk_name))
        conn.execute('INSERT INTO page_vector (page_id, embedding_vector) VALUES (?, ?)', 
                    (i, vector.tobytes()))

    conn.commit()

    # Test with a very small batch size that would normally cause the error
    try:
        batch_count, total_vectors = run_pca(conn, namespace, target_dim=100, batch_size=10)
        print(f'Success with small batch size! Processed {total_vectors} vectors in {batch_count} batches')
        
        # Verify that reduced vectors were stored
        cursor = conn.execute('SELECT COUNT(*) as count FROM page_vector WHERE reduced_vector IS NOT NULL')
        result = cursor.fetchone()
        print(f'Reduced vectors stored: {result["count"]}')
        
    # except Exception as e:
    #     print(f'Error: {e}')
    #     import traceback
    #     traceback.print_exc()

    finally:
        conn.close()


def test_kmeans_basic():
    """Test basic K-means clustering functionality with synthetic reduced vectors."""
    # Create a test database with proper schema
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Ensure tables are created with proper schema
    ensure_tables(conn)

    # Insert test data - create reduced vectors (100-dimensional) for K-means
    namespace = 'test_namespace'
    chunk_name = 'test_chunk'
    conn.execute('INSERT INTO chunk_log (chunk_name, namespace) VALUES (?, ?)',
                (chunk_name, namespace))
    
    # Create 150 reduced vectors of 100 dimensions each
    n_vectors = 150
    target_dim = 100
    for i in range(n_vectors):
        # Create synthetic reduced vectors (100-dimensional)
        reduced_vector = np.random.randn(target_dim).astype(np.float32)
        conn.execute('INSERT INTO page_log (page_id, chunk_name) VALUES (?, ?)',
                     (i, chunk_name))
        conn.execute('INSERT INTO page_vector (page_id, embedding_vector, reduced_vector) VALUES (?, ?, ?)',
                    (i, np.random.randn(2048).astype(np.float32).tobytes(), reduced_vector.tobytes()))

    conn.commit()

    # Test the run_kmeans function
    try:
        run_kmeans(conn, namespace, n_clusters=10, batch_size=50)
        print(f'Success! K-means clustering completed')
        
        # Verify that cluster assignments were stored
        cursor = conn.execute('SELECT COUNT(*) as count FROM page_vector WHERE cluster_id IS NOT NULL')
        result = cursor.fetchone()
        print(f'Pages with cluster assignments: {result["count"]}')
        
        # Verify that cluster_info entries were created
        cursor = conn.execute('SELECT COUNT(*) as count FROM cluster_info WHERE namespace = ?', (namespace,))
        result = cursor.fetchone()
        print(f'Cluster info entries created: {result["count"]}')
        
        # Verify that all pages have been assigned to clusters
        cursor = conn.execute('SELECT MIN(cluster_id) as min_cluster, MAX(cluster_id) as max_cluster FROM page_vector WHERE cluster_id IS NOT NULL')
        result = cursor.fetchone()
        print(f'Cluster range: {result["min_cluster"]} to {result["max_cluster"]}')
        
        # Check that all expected clusters have at least one page
        cursor = conn.execute('SELECT cluster_id, COUNT(*) as count FROM page_vector WHERE cluster_id IS NOT NULL GROUP BY cluster_id ORDER BY cluster_id')
        cluster_assignments = cursor.fetchall()
        print(f'Cluster assignments: {[(row["cluster_id"], row["count"]) for row in cluster_assignments]}')
        cluster_id_set = set()
        for cluster in cluster_assignments:
            assert cluster[1] > 0, f"Empty cluster found: {cluster[0]}"
            assert cluster[0] not in cluster_id_set, f"Duplicate cluster ID found: {cluster[0]}"
            cluster_id_set.add(cluster[0])

    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


def test_kmeans_edge_cases():
    """Test K-means with edge cases like small number of vectors and clusters."""
    # Create a test database with proper schema
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Ensure tables are created with proper schema
    ensure_tables(conn)

    # Insert test data - create only 5 reduced vectors
    namespace = 'test_namespace_small'
    chunk_name = 'test_chunk_small'
    conn.execute('INSERT INTO chunk_log (chunk_name, namespace) VALUES (?, ?)',
                (chunk_name, namespace))
    
    # Create only 5 reduced vectors
    n_vectors = 5
    target_dim = 100
    for i in range(n_vectors):
        reduced_vector = np.random.randn(target_dim).astype(np.float32)
        conn.execute('INSERT INTO page_log (page_id, chunk_name) VALUES (?, ?)',
                     (i, chunk_name))
        conn.execute('INSERT INTO page_vector (page_id, embedding_vector, reduced_vector) VALUES (?, ?, ?)',
                    (i, np.random.randn(2048).astype(np.float32).tobytes(), reduced_vector.tobytes()))

    conn.commit()

    # Test with fewer clusters than vectors (more realistic scenario)
    try:
        run_kmeans(conn, namespace, n_clusters=3, batch_size=3)
        print(f'Success! K-means with fewer clusters than vectors completed')
        
        # Verify that cluster assignments were stored
        cursor = conn.execute('SELECT COUNT(*) as count FROM page_vector WHERE cluster_id IS NOT NULL')
        result = cursor.fetchone()
        print(f'Pages with cluster assignments: {result["count"]}')
        
        # Verify that cluster_info entries were created
        cursor = conn.execute('SELECT COUNT(*) as count FROM cluster_info WHERE namespace = ?', (namespace,))
        result = cursor.fetchone()
        print(f'Cluster info entries created: {result["count"]}')
        
        # Check distribution across clusters
        cursor = conn.execute('SELECT cluster_id, COUNT(*) as count FROM page_vector WHERE cluster_id IS NOT NULL GROUP BY cluster_id ORDER BY cluster_id')
        cluster_assignments = cursor.fetchall()
        print(f'Cluster assignments: {[(row["cluster_id"], row["count"]) for row in cluster_assignments]}')
        cluster_id_set = set()
        for cluster in cluster_assignments:
            assert cluster[1] > 0, f"Empty cluster found: {cluster[0]}"
            assert cluster[0] not in cluster_id_set, f"Duplicate cluster ID found: {cluster[0]}"
            cluster_id_set.add(cluster[0])
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


def test_kmeans_no_reduced_vectors():
    """Test K-means when no reduced vectors are available."""
    # Create a test database with proper schema
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Ensure tables are created with proper schema
    ensure_tables(conn)

    # Insert test data - create only original embedding vectors (no reduced vectors)
    namespace = 'test_namespace_no_reduced'
    chunk_name = 'test_chunk_no_reduced'
    conn.execute('INSERT INTO chunk_log (chunk_name, namespace) VALUES (?, ?)',
                (chunk_name, namespace))
    
    # Create only original embedding vectors (no reduced vectors)
    n_vectors = 10
    for i in range(n_vectors):
        conn.execute('INSERT INTO page_log (page_id, chunk_name) VALUES (?, ?)',
                     (i, chunk_name))
        conn.execute('INSERT INTO page_vector (page_id, embedding_vector) VALUES (?, ?)',
                    (i, np.random.randn(2048).astype(np.float32).tobytes()))

    conn.commit()

    # Test K-means with no reduced vectors (should handle gracefully)
    try:
        run_kmeans(conn, namespace, n_clusters=5, batch_size=5)
        print('K-means completed with no reduced vectors')
        
        # Verify that no cluster assignments were made
        cursor = conn.execute('SELECT COUNT(*) as count FROM page_vector WHERE cluster_id IS NOT NULL')
        result = cursor.fetchone()
        print(f'Pages with cluster assignments: {result["count"]}')
        
        # Verify that no cluster_info entries were created
        cursor = conn.execute('SELECT COUNT(*) as count FROM cluster_info WHERE namespace = ?', (namespace,))
        result = cursor.fetchone()
        print(f'Cluster info entries created: {result["count"]}')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()



def test_kmeans_centroids():
    """Test that K-means centroids are properly stored as BLOB data."""
    # Create a test database with proper schema
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Ensure tables are created with proper schema
    ensure_tables(conn)

    # Insert test data - create reduced vectors (100-dimensional) for K-means
    namespace = 'test_namespace_centroids'
    chunk_name = 'test_chunk_centroids'
    conn.execute('INSERT INTO chunk_log (chunk_name, namespace) VALUES (?, ?)',
                (chunk_name, namespace))
    
    # Create 200 reduced vectors of 100 dimensions each with some structure
    n_vectors = 200
    target_dim = 100
    n_clusters = 5
    
    # Create structured data with clear cluster centers
    cluster_centers = []
    for cluster_id in range(n_clusters):
        # Each cluster has a distinct center
        center = np.random.randn(target_dim).astype(np.float32) * 2  # Scale for separation
        cluster_centers.append(center)
        
        # Create vectors around this center with some noise
        for i in range(n_vectors // n_clusters):
            noise = np.random.randn(target_dim).astype(np.float32) * 0.1
            vector = center + noise
            page_id = cluster_id * (n_vectors // n_clusters) + i
            conn.execute('INSERT INTO page_log (page_id, chunk_name) VALUES (?, ?)',
                         (page_id, chunk_name))
            conn.execute('INSERT INTO page_vector (page_id, embedding_vector, reduced_vector) VALUES (?, ?, ?)',
                         (page_id, np.random.randn(2048).astype(np.float32).tobytes(), vector.tobytes()))

    conn.commit()

    # Test the run_kmeans function
    try:
        run_kmeans(conn, namespace, n_clusters=n_clusters, batch_size=50)
        print(f'Success! K-means clustering completed with {n_clusters} clusters')
        
        # Verify that cluster centroids were stored as BLOB
        cursor = conn.execute('SELECT COUNT(*) as count FROM cluster_info WHERE centroid_vector IS NOT NULL AND namespace = ?', (namespace,))
        result = cursor.fetchone()
        print(f'Clusters with centroids stored: {result["count"]}')
        
        # Verify that all expected clusters have centroids
        assert result["count"] == n_clusters, f"Expected {n_clusters} clusters with centroids, got {result['count']}"
        
        # Fetch and verify each centroid
        cursor = conn.execute('SELECT cluster_id, centroid_vector FROM cluster_info WHERE namespace = ? ORDER BY cluster_id', (namespace,))
        stored_centroids = cursor.fetchall()
        
        for row in stored_centroids:
            cluster_id = row["cluster_id"]
            centroid_blob = row["centroid_vector"]
            
            # Convert BLOB back to numpy array
            stored_centroid = np.frombuffer(centroid_blob, dtype=np.float32)
            
            # Verify centroid has correct dimensionality
            assert len(stored_centroid) == target_dim, f"Centroid {cluster_id} has wrong dimensionality: {len(stored_centroid)}, expected {target_dim}"
            
            # Verify centroid is not all zeros (basic sanity check)
            assert not np.allclose(stored_centroid, 0), f"Centroid {cluster_id} is all zeros"
            
            print(f'Cluster {cluster_id}: centroid shape={stored_centroid.shape}')
        
        # Verify that cluster assignments exist and are consistent with centroids
        cursor = conn.execute('SELECT cluster_id, COUNT(*) as count FROM page_vector WHERE cluster_id IS NOT NULL GROUP BY cluster_id ORDER BY cluster_id')
        cluster_assignments = cursor.fetchall()
        
        for assignment in cluster_assignments:
            cluster_id = assignment["cluster_id"]
            count = assignment["count"]
            
            # Verify each cluster has at least one assignment
            assert count > 0, f"Cluster {cluster_id} has no assignments"
            
            # Verify centroid exists for this cluster
            cursor = conn.execute('SELECT 1 FROM cluster_info WHERE cluster_id = ? AND namespace = ? AND centroid_vector IS NOT NULL', (cluster_id, namespace))
            assert cursor.fetchone(), f"No centroid found for cluster {cluster_id}"
            
            print(f'Cluster {cluster_id}: {count} pages assigned')
        
        print('All centroid storage assertions passed!')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print("Running PCA tests...")
    test_pca_batch_sufficient()
    test_pca_insufficient_batch()
    print("\nRunning K-means tests...")
    test_kmeans_basic()
    test_kmeans_edge_cases()
    test_kmeans_no_reduced_vectors()
    test_kmeans_centroids()
    print("\nAll tests completed!")
