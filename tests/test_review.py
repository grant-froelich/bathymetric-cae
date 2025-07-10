# tests/test_review.py
"""Tests for expert review system."""

import pytest
import tempfile
from pathlib import Path

from bathymetric_cae.review import ExpertReviewSystem, ReviewDatabase
from bathymetric_cae.core.enums import ExpertReviewFlag


class TestReviewDatabase:
    """Test review database functionality."""
    
    def test_database_creation(self):
        """Test database initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            db = ReviewDatabase(db_path)
            assert Path(db_path).exists()
            
            # Test database info
            info = db.get_database_info()
            assert 'file_size_mb' in info
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_flag_and_review_workflow(self):
        """Test complete flag and review workflow."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            db = ReviewDatabase(db_path)
            
            # Flag a region
            flag_id = db.flag_region_for_review(
                "test_file.bag",
                (0, 0, 100, 100),
                ExpertReviewFlag.LOW_QUALITY.value,
                0.3,
                "Test flagging"
            )
            
            assert flag_id > 0
            
            # Check pending reviews
            pending = db.get_pending_reviews()
            assert len(pending) == 1
            assert pending[0]['filename'] == "test_file.bag"
            
            # Submit review
            review_id = db.submit_expert_review(
                "test_file.bag",
                str(flag_id),
                3,  # rating
                0.7,  # quality score
                "Test review",
                "test_reviewer"
            )
            
            assert review_id > 0
            
            # Check review statistics
            stats = db.get_review_statistics()
            assert stats['total_reviews'] == 1
            assert stats['pending_reviews'] == 0
            
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestExpertReviewSystem:
    """Test expert review system."""
    
    def test_system_creation(self):
        """Test system initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            system = ExpertReviewSystem(db_path)
            assert hasattr(system, 'db')
            assert hasattr(system, 'auto_review_thresholds')
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_auto_quality_review(self):
        """Test automatic quality review."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            system = ExpertReviewSystem(db_path)
            
            # Test excellent quality (should auto-approve)
            excellent_metrics = {'composite_quality': 0.95}
            review_id = system.auto_review_quality("excellent_file.bag", excellent_metrics)
            assert review_id is not None
            
            # Test poor quality (should auto-flag)
            poor_metrics = {'composite_quality': 0.3}
            review_id = system.auto_review_quality("poor_file.bag", poor_metrics)
            assert review_id is None  # Flagged, not reviewed
            
            # Check that poor file was flagged
            pending = system.get_pending_reviews()
            assert len(pending) > 0
            
        finally:
            Path(db_path).unlink(missing_ok=True)