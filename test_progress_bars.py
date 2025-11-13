#!/usr/bin/env python3
"""
Test script to verify progress bars work correctly using pytest.
"""

import sys
import os
import tempfile
import json
import time
from progress_utils import ProgressTracker

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_progress_tracker_basic():
    """Test the ProgressTracker with basic progress tracking."""
    # Test basic progress tracking with total
    tracker_ref = None
    with ProgressTracker("Test operation", total=10, unit="items") as tracker:
        tracker_ref = tracker
        # Verify the tracker was created successfully
        assert tracker.progress_bar is not None
        assert tracker.description == "Test operation"
        assert tracker.total == 10
        assert tracker.unit == "items"
        assert tracker._is_started is True

        for i in range(10):
            time.sleep(0.01)  # Reduced sleep for faster testing
            tracker.update(1)

    # After context exit, tracker should be closed
    assert tracker_ref.progress_bar is None
    assert tracker_ref._is_started is False


def test_progress_tracker_without_total():
    """Test the ProgressTracker without a total value."""
    # Test progress without total
    tracker_ref = None
    with ProgressTracker("Indefinite operation", unit="steps") as tracker:
        tracker_ref = tracker
        # Verify the tracker was created successfully
        assert tracker.progress_bar is not None
        assert tracker.description == "Indefinite operation"
        assert tracker.total is None
        assert tracker.unit == "steps"
        assert tracker._is_started is True

        for i in range(5):
            time.sleep(0.01)
            tracker.update(1)

    # After context exit, tracker should be closed
    assert tracker_ref.progress_bar is None
    assert tracker_ref._is_started is False


def test_progress_tracker_large_numbers():
    """Test the ProgressTracker with large numbers."""
    # Test with large numbers
    tracker_ref = None
    with ProgressTracker("Large operation", total=1000, unit="items") as tracker:
        tracker_ref = tracker
        # Verify the tracker was created successfully
        assert tracker.progress_bar is not None
        assert tracker.description == "Large operation"
        assert tracker.total == 1000
        assert tracker.unit == "items"
        assert tracker._is_started is True

        for i in range(0, 1000, 50):
            time.sleep(0.001)  # Reduced sleep for faster testing
            tracker.update(50)

    # After context exit, tracker should be closed
    assert tracker_ref.progress_bar is None
    assert tracker_ref._is_started is False


def test_progress_tracker_manual_operations():
    """Test manual progress tracker operations."""
    tracker = ProgressTracker("Manual test", total=100, unit="items")

    # Test that tracker starts correctly
    tracker.start()
    assert tracker.progress_bar is not None
    assert tracker._is_started is True

    # Test increment operation
    tracker.increment(10)
    assert tracker.progress_bar.n == 10

    # Test set_total operation
    tracker.set_total(200)
    assert tracker.total == 200

    # Test set_progress operation
    tracker.set_progress(50)
    assert tracker.progress_bar.n == 50

    # Test close operation
    tracker.close()
    assert tracker.progress_bar is None
    assert tracker._is_started is False


def test_progress_tracker_context_manager():
    """Test that ProgressTracker works correctly as a context manager."""
    tracker = None

    with ProgressTracker("Context test", total=5, unit="items") as t:
        tracker = t
        assert tracker.progress_bar is not None
        assert tracker._is_started is True

        for i in range(5):
            time.sleep(0.01)
            tracker.update(1)

    # After context exit, tracker should be closed
    assert tracker.progress_bar is None
    assert tracker._is_started is False


def test_chunk_parsing_simulation():
    """Simulate the chunk parsing operation with progress."""
    # Create a temporary JSON file with sample data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name
        # Write some sample JSON lines
        for i in range(50):  # Reduced from 500 for faster testing
            sample_data = {
                "id": i + 1,
                "name": f"Sample Page {i + 1}",
                "url": f"https://example.com/page/{i + 1}",
                "abstract": f"This is sample abstract text for page {i + 1}. " * 3,
            }
            f.write(json.dumps(sample_data) + "\n")

    try:
        # Count lines first (as in the real implementation)
        total_lines = 0
        with open(temp_file, "r") as f:
            for _ in f:
                total_lines += 1

        assert total_lines == 50  # Verify we created the expected number of lines

        # Parse with progress bar
        tracker_ref = None
        with ProgressTracker(
            "Parsing sample chunk", total=total_lines, unit="lines"
        ) as tracker:
            tracker_ref = tracker
            # Verify tracker is active during processing
            assert tracker.progress_bar is not None
            assert tracker._is_started is True

            with open(temp_file, "r") as f:
                for line in f:
                    # Simulate processing
                    data = json.loads(line)
                    assert data["id"] is not None  # Verify JSON parsing worked
                    time.sleep(0.001)  # Reduced sleep for faster testing
                    tracker.update(1)

        # After context exit, tracker should be closed
        assert tracker_ref.progress_bar is None
        assert tracker_ref._is_started is False

    finally:
        # Clean up
        os.unlink(temp_file)


def test_embedding_computation_simulation():
    """Simulate the embedding computation operation with progress."""
    # Simulate processing pages for embeddings
    num_pages = 20  # Reduced from 200 for faster testing
    assert num_pages > 0

    tracker_ref = None
    with ProgressTracker(
        "Computing embeddings", total=num_pages, unit="pages"
    ) as tracker:
        tracker_ref = tracker
        # Verify tracker is active during processing
        assert tracker.progress_bar is not None
        assert tracker._is_started is True

        for i in range(num_pages):
            # Simulate embedding computation time
            time.sleep(0.01)  # Reduced sleep for faster testing
            tracker.update(1)

    # After context exit, tracker should be closed
    assert tracker_ref.progress_bar is None
    assert tracker_ref._is_started is False


def test_progress_tracker_exception_handling():
    """Test that ProgressTracker handles exceptions correctly."""
    tracker = ProgressTracker("Exception test", total=10, unit="items")

    try:
        with tracker:
            assert tracker.progress_bar is not None
            tracker.update(5)
            # Simulate an exception
            raise ValueError("Test exception")
    except ValueError:
        pass  # Expected exception

    # Verify tracker is closed even after exception
    assert tracker.progress_bar is None
    assert tracker._is_started is False


def test_progress_tracker_multiple_instances():
    """Test that multiple ProgressTracker instances work independently."""
    tracker1 = ProgressTracker("Tracker 1", total=10, unit="items")
    tracker2 = ProgressTracker("Tracker 2", total=20, unit="steps")

    # Start both trackers
    tracker1.start()
    tracker2.start()

    assert tracker1.progress_bar is not None
    assert tracker2.progress_bar is not None
    assert tracker1 is not tracker2

    # Update both trackers
    tracker1.update(5)
    tracker2.update(10)

    # Close both trackers
    tracker1.close()
    tracker2.close()

    assert tracker1.progress_bar is None
    assert tracker2.progress_bar is None
