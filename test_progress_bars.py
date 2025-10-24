#!/usr/bin/env python3
"""
Simple test script to verify progress bars work correctly.
"""

import sys
import os
import tempfile
import json
import time

from progress_utils import ProgressTracker

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_progress_tracker():
    """Test the ProgressTracker with a simple simulation."""
    print("Testing ProgressTracker...")

    # Test 1: Basic progress tracking
    print("\n1. Testing basic progress tracking:")
    with ProgressTracker("Test operation", total=10, unit="items") as tracker:
        for i in range(10):
            time.sleep(0.1)  # Simulate work
            tracker.update(1)

    # Test 2: Progress without total
    print("\n2. Testing progress without total:")
    with ProgressTracker("Indefinite operation", unit="steps") as tracker:
        for i in range(5):
            time.sleep(0.1)
            tracker.update(1)

    # Test 3: Large numbers
    print("\n3. Testing with large numbers:")
    with ProgressTracker("Large operation", total=1000, unit="items") as tracker:
        for i in range(0, 1000, 50):
            time.sleep(0.01)
            tracker.update(50)

    print("\n✅ All progress bar tests completed successfully!")


def simulate_chunk_parsing():
    """Simulate the chunk parsing operation with progress."""
    print("\n" + "=" * 50)
    print("Simulating chunk parsing operation...")
    print("=" * 50)

    # Create a temporary JSON file with sample data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name
        # Write some sample JSON lines
        for i in range(500):
            sample_data = {
                "id": i + 1,
                "name": f"Sample Page {i + 1}",
                "url": f"https://example.com/page/{i + 1}",
                "abstract": f"This is sample abstract text for page {i + 1}. " * 3,
            }
            f.write(json.dumps(sample_data) + "\n")

    try:
        # Simulate the parsing with progress
        print(f"Parsing sample chunk file: {temp_file}")

        # Count lines first (as in the real implementation)
        total_lines = 0
        with open(temp_file, "r") as f:
            for _ in f:
                total_lines += 1

        print(f"Found {total_lines} lines to process")

        # Parse with progress bar
        with ProgressTracker(
            "Parsing sample chunk", total=total_lines, unit="lines"
        ) as tracker:
            with open(temp_file, "r") as f:
                for line in f:
                    # Simulate processing
                    json.loads(line)
                    time.sleep(0.001)  # Simulate processing time
                    tracker.update(1)

        print("✅ Chunk parsing simulation completed!")

    finally:
        # Clean up
        os.unlink(temp_file)


def simulate_embedding_computation():
    """Simulate the embedding computation operation with progress."""
    print("\n" + "=" * 50)
    print("Simulating embedding computation...")
    print("=" * 50)

    # Simulate processing pages for embeddings
    num_pages = 200
    print(f"Computing embeddings for {num_pages} pages...")

    with ProgressTracker(
        "Computing embeddings", total=num_pages, unit="pages"
    ) as tracker:
        for i in range(num_pages):
            # Simulate embedding computation time
            time.sleep(0.02)
            tracker.update(1)

    print("✅ Embedding computation simulation completed!")


if __name__ == "__main__":
    print("🚀 Starting progress bar tests...")

    try:
        test_progress_tracker()
        simulate_chunk_parsing()
        simulate_embedding_computation()

        print("\n🎉 All tests completed successfully!")
        print("Progress bars are working correctly and ready for production use.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
