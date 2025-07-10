# examples/expert_review_example.py
"""
Expert review system example for Enhanced Bathymetric CAE Processing.

This example demonstrates how to use the expert review system for
quality control and human-in-the-loop validation.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bathymetric_cae.review import ExpertReviewSystem
from bathymetric_cae.core.enums import ExpertReviewFlag
from bathymetric_cae.utils import setup_logging


def setup_review_system():
    """Initialize the expert review system."""
    
    # Create example database
    db_path = "examples/example_reviews.db"
    
    # Initialize review system
    reviewer = ExpertReviewSystem(db_path)
    
    print(f"Expert Review System initialized with database: {db_path}")
    return reviewer


def demonstrate_flagging_workflow(reviewer):
    """Demonstrate the flagging workflow."""
    
    print("\n1. Flagging Files for Review")
    print("-" * 30)
    
    # Example 1: Flag a file with low quality
    flag_id_1 = reviewer.flag_for_review(
        filename="shallow_survey_001.bag",
        region=(0, 0, 512, 512),
        flag_type=ExpertReviewFlag.LOW_QUALITY.value,
        confidence=0.3,
        description="Low SSIM score (0.45) in shallow water region"
    )
    print(f"✓ Flagged file for low quality (ID: {flag_id_1})")
    
    # Example 2: Flag a file with feature loss
    flag_id_2 = reviewer.flag_for_review(
        filename="deep_trench_002.bag", 
        region=(100, 100, 400, 400),
        flag_type=ExpertReviewFlag.FEATURE_LOSS.value,
        confidence=0.8,
        description="Significant bathymetric features lost during processing"
    )
    print(f"✓ Flagged file for feature loss (ID: {flag_id_2})")
    
    # Example 3: Flag for standards violation
    flag_id_3 = reviewer.flag_for_review(
        filename="coastal_mapping_003.bag",
        region=(200, 200, 300, 300),
        flag_type=ExpertReviewFlag.STANDARDS_VIOLATION.value,
        confidence=0.9,
        description="Does not meet IHO Order 1a standards"
    )
    print(f"✓ Flagged file for standards violation (ID: {flag_id_3})")
    
    return [flag_id_1, flag_id_2, flag_id_3]


def demonstrate_review_submission(reviewer, flag_ids):
    """Demonstrate submitting expert reviews."""
    
    print("\n2. Submitting Expert Reviews")
    print("-" * 30)
    
    # Review 1: Approve with minor concerns
    review_id_1 = reviewer.submit_review(
        filename="shallow_survey_001.bag",
        region_id=str(flag_ids[0]),
        rating=3,  # Acceptable
        quality_score=0.7,
        comments="Quality acceptable after manual inspection. Some noise in shallow areas but overall usable.",
        reviewer_id="expert_hydrographer_1"
    )
    print(f"✓ Submitted review for shallow survey (ID: {review_id_1})")
    
    # Review 2: Reject and request reprocessing
    review_id_2 = reviewer.submit_review(
        filename="deep_trench_002.bag",
        region_id=str(flag_ids[1]),
        rating=2,  # Poor
        quality_score=0.4,
        comments="Critical bathymetric features lost. Recommend reprocessing with higher feature preservation weight.",
        reviewer_id="expert_marine_geologist"
    )
    print(f"✓ Submitted review for deep trench (ID: {review_id_2})")
    
    # Review 3: Escalate for further review
    review_id_3 = reviewer.submit_review(
        filename="coastal_mapping_003.bag",
        region_id=str(flag_ids[2]),
        rating=2,  # Poor  
        quality_score=0.45,
        comments="Standards violation confirmed. Escalating to senior hydrographer for final decision.",
        reviewer_id="expert_coastal_specialist"
    )
    print(f"✓ Submitted review for coastal mapping (ID: {review_id_3})")
    
    return [review_id_1, review_id_2, review_id_3]


def demonstrate_auto_review(reviewer):
    """Demonstrate automatic quality review."""
    
    print("\n3. Automatic Quality Review")
    print("-" * 30)
    
    # Example 1: Excellent quality (auto-approve)
    excellent_metrics = {
        'composite_quality': 0.95,
        'ssim': 0.93,
        'feature_preservation': 0.97,
        'consistency': 0.94
    }
    
    auto_review_1 = reviewer.auto_review_quality(
        "excellent_survey.bag", 
        excellent_metrics
    )
    
    if auto_review_1:
        print("✓ Excellent quality file auto-approved")
    else:
        print("✗ Auto-approval failed")
    
    # Example 2: Poor quality (auto-flag)
    poor_metrics = {
        'composite_quality': 0.35,
        'ssim': 0.4,
        'feature_preservation': 0.3,
        'consistency': 0.35
    }
    
    auto_review_2 = reviewer.auto_review_quality(
        "poor_quality_survey.bag",
        poor_metrics  
    )
    
    if auto_review_2 is None:
        print("✓ Poor quality file auto-flagged for review")
    else:
        print("✗ Auto-flagging failed")


def demonstrate_review_queue_management(reviewer):
    """Demonstrate review queue management."""
    
    print("\n4. Review Queue Management")
    print("-" * 30)
    
    # Get pending reviews
    pending = reviewer.get_pending_reviews(limit=10)
    print(f"Pending reviews: {len(pending)}")
    
    if pending:
        print("\nTop priority items:")
        for item in pending[:3]:
            print(f"  - {item['filename']} (Priority: {item['priority']}, "
                  f"Type: {item['flag_type']}, Age: {item.get('age_days', 0)} days)")
    
    # Get review queue for specific reviewer
    queue = reviewer.get_review_queue("expert_hydrographer_1", max_items=5)
    print(f"\nItems assigned to expert_hydrographer_1: {len(queue)}")
    
    # Assign reviewer to pending item
    if pending:
        success = reviewer.assign_reviewer(pending[0]['id'], "expert_senior_hydrographer")
        if success:
            print("✓ Assigned senior hydrographer to high-priority item")


def demonstrate_reporting(reviewer):
    """Demonstrate review reporting."""
    
    print("\n5. Review Reporting")
    print("-" * 30)
    
    # Generate review report
    report = reviewer.generate_review_report(days=30)
    
    print("Review Report Summary:")
    print(f"  Total reviews: {report['statistics'].get('total_reviews', 0)}")
    print(f"  Pending reviews: {report['statistics'].get('pending_reviews', 0)}")
    print(f"  Average quality: {report['statistics'].get('average_quality_score', 0):.3f}")
    
    # Print recommendations
    if 'recommendations' in report:
        print("\nRecommendations:")
        for rec in report['recommendations'][:3]:
            print(f"  • {rec}")
    
    # Get reviewer dashboard
    dashboard = reviewer.get_reviewer_dashboard("expert_hydrographer_1")
    
    print(f"\nReviewer Dashboard (expert_hydrographer_1):")
    print(f"  Assigned items: {dashboard.get('assigned_items', 0)}")
    print(f"  High priority items: {dashboard.get('high_priority_items', 0)}")
    print(f"  Completed reviews: {dashboard['workload'].get('completed_reviews', 0)}")


def demonstrate_batch_operations(reviewer):
    """Demonstrate batch operations."""
    
    print("\n6. Batch Operations")
    print("-" * 30)
    
    # Batch approve reviews meeting criteria
    approval_criteria = {
        'min_quality_score': 0.8,
        'max_age_days': 7
    }
    
    approved_count = reviewer.batch_approve_reviews(
        criteria=approval_criteria,
        approver_id="senior_expert"
    )
    
    print(f"✓ Batch approved {approved_count} reviews")
    
    # Export review data
    try:
        export_path = reviewer.export_review_data("examples/", format='csv')
        print(f"✓ Exported review data to: {export_path}")
    except Exception as e:
        print(f"✗ Export failed: {e}")


def main():
    """Run the complete expert review example."""
    
    print("Expert Review System Example")
    print("=" * 40)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Initialize review system
    reviewer = setup_review_system()
    
    try:
        # Run through all examples
        flag_ids = demonstrate_flagging_workflow(reviewer)
        review_ids = demonstrate_review_submission(reviewer, flag_ids)
        demonstrate_auto_review(reviewer)
        demonstrate_review_queue_management(reviewer)
        demonstrate_reporting(reviewer)
        demonstrate_batch_operations(reviewer)
        
        print("\n" + "=" * 40)
        print("Expert Review Example Completed Successfully!")
        print("\nDatabase file: examples/example_reviews.db")
        print("You can inspect the database or integrate this workflow into your processing pipeline.")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
