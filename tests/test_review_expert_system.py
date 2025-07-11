# tests/test_review_expert_system.py
"""
Test expert review system.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from review.expert_system import ExpertReviewSystem


class TestExpertReviewSystem:
    """Test expert review system functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)
    
    def test_database_setup(self, temp_db):
        """Test database setup."""
        review_system = ExpertReviewSystem(temp_db)
        
        # Check that tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        assert 'expert_reviews' in tables
        assert 'flagged_regions' in tables
        
        conn.close()
    
    def test_flag_for_review(self, temp_db):
        """Test flagging regions for review."""
        review_system = ExpertReviewSystem(temp_db)
        
        # Flag a region
        review_system.flag_for_review(
            "test_file.bag", (0, 0, 100, 100), "low_quality", 0.8
        )
        
        # Check that it was flagged
        pending = review_system.get_pending_reviews()
        assert len(pending) == 1
        assert pending[0]['filename'] == "test_file.bag"
        assert pending[0]['flag_type'] == "low_quality"
    
    def test_submit_review(self, temp_db):
        """Test submitting expert review."""
        review_system = ExpertReviewSystem(temp_db)
        
        # Submit a review
        review_system.submit_review(
            "test_file.bag", "region_1", 3, 0.75, "Good quality overall"
        )
        
        # Verify review was submitted
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM expert_reviews")
        reviews = cursor.fetchall()
        conn.close()
        
        assert len(reviews) == 1
        assert reviews[0][1] == "test_file.bag"  # filename
        assert reviews[0][3] == 3  # expert_rating
    
    def test_mark_review_complete(self, temp_db):
        """Test marking review as complete."""
        review_system = ExpertReviewSystem(temp_db)
        
        # Flag and then mark as complete
        review_system.flag_for_review(
            "test_file.bag", (0, 0, 100, 100), "low_quality", 0.8
        )
        
        # Get the region ID
        pending = review_system.get_pending_reviews()
        region_id = pending[0]['id']
        
        # Mark as complete
        review_system.mark_review_complete(region_id)
        
        # Check that it's no longer pending
        pending_after = review_system.get_pending_reviews()
        assert len(pending_after) == 0
    
    def test_review_statistics(self, temp_db):
        """Test review statistics calculation."""
        review_system = ExpertReviewSystem(temp_db)
        
        # Flag multiple regions
        for i in range(3):
            review_system.flag_for_review(
                f"test_file_{i}.bag", (0, 0, 100, 100), "low_quality", 0.8
            )
        
        # Mark one as complete
        pending = review_system.get_pending_reviews()
        review_system.mark_review_complete(pending[0]['id'])
        
        # Get statistics
        stats = review_system.get_review_statistics()
        
        assert stats['total_flagged'] == 3
        assert stats['total_reviewed'] == 1
        assert stats['pending_reviews'] == 2
        assert stats['completion_rate'] == pytest.approx(33.33, abs=0.1)