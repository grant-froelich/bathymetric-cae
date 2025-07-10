"""
Expert review system for Enhanced Bathymetric CAE Processing.

This module implements the human-in-the-loop validation system for quality control
and expert oversight of bathymetric data processing results.
"""

import logging
import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from ..core.enums import ExpertReviewFlag, QualityLevel
from .database import ReviewDatabase


class ExpertReviewSystem:
    """Human-in-the-loop validation system for quality control."""
    
    def __init__(self, db_path: str = "expert_reviews.db"):
        self.db = ReviewDatabase(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Review configuration
        self.auto_review_thresholds = {
            'excellent_threshold': 0.9,
            'acceptable_threshold': 0.7,
            'poor_threshold': 0.5
        }
        
        # Flag priorities
        self.flag_priorities = {
            ExpertReviewFlag.STANDARDS_VIOLATION: 1,
            ExpertReviewFlag.PROCESSING_ERROR: 1,
            ExpertReviewFlag.FEATURE_LOSS: 2,
            ExpertReviewFlag.HIGH_UNCERTAINTY: 2,
            ExpertReviewFlag.LOW_QUALITY: 3,
            ExpertReviewFlag.POOR_SIMILARITY: 3,
            ExpertReviewFlag.MANUAL_REQUEST: 4
        }
        
        self.logger.info("Expert Review System initialized")
    
    def flag_for_review(self, filename: str, region: Tuple[int, int, int, int], 
                       flag_type: str, confidence: float, 
                       description: str = "") -> int:
        """Flag a region for expert review."""
        try:
            # Validate flag type
            if flag_type not in [flag.value for flag in ExpertReviewFlag]:
                raise ValueError(f"Invalid flag type: {flag_type}")
            
            # Enhanced description with context
            enhanced_description = self._enhance_flag_description(flag_type, confidence, description)
            
            # Flag in database
            flag_id = self.db.flag_region_for_review(
                filename, region, flag_type, confidence, enhanced_description
            )
            
            # Log system event
            self.db.log_system_event(
                "region_flagged",
                {
                    "filename": filename,
                    "flag_type": flag_type,
                    "confidence": confidence,
                    "flag_id": flag_id
                }
            )
            
            self.logger.info(f"Flagged {filename} for {flag_type} review (confidence: {confidence:.3f})")
            
            return flag_id
            
        except Exception as e:
            self.logger.error(f"Error flagging region for review: {e}")
            raise
    
    def _enhance_flag_description(self, flag_type: str, confidence: float, description: str) -> str:
        """Enhance flag description with additional context."""
        flag_enum = ExpertReviewFlag(flag_type)
        
        enhanced = f"{flag_enum.description} (Confidence: {confidence:.3f})"
        
        if description:
            enhanced += f" - {description}"
        
        # Add priority context
        priority = flag_enum.priority
        priority_desc = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Info"}
        enhanced += f" [Priority: {priority_desc.get(priority, 'Unknown')}]"
        
        return enhanced
    
    def get_pending_reviews(self, limit: Optional[int] = None, 
                           priority_filter: Optional[int] = None) -> List[Dict]:
        """Get regions pending expert review with enhanced filtering."""
        try:
            pending = self.db.get_pending_reviews(limit, priority_filter)
            
            # Enhance with additional context
            for review in pending:
                review['priority_description'] = self._get_priority_description(review.get('priority', 3))
                review['flag_description'] = self._get_flag_description(review.get('flag_type', ''))
                review['age_days'] = self._calculate_age_days(review.get('flagged_date'))
            
            return pending
            
        except Exception as e:
            self.logger.error(f"Error getting pending reviews: {e}")
            return []
    
    def _get_priority_description(self, priority: int) -> str:
        """Get human-readable priority description."""
        descriptions = {
            1: "Critical - Immediate attention required",
            2: "High - Review within 24 hours", 
            3: "Medium - Review within 3 days",
            4: "Low - Review when convenient",
            5: "Info - For reference only"
        }
        return descriptions.get(priority, "Unknown priority")
    
    def _get_flag_description(self, flag_type: str) -> str:
        """Get detailed flag description."""
        try:
            flag_enum = ExpertReviewFlag(flag_type)
            return flag_enum.description
        except ValueError:
            return "Unknown flag type"
    
    def _calculate_age_days(self, flagged_date: str) -> int:
        """Calculate age of flagged item in days."""
        try:
            if not flagged_date:
                return 0
            
            flagged_dt = datetime.datetime.fromisoformat(flagged_date.replace('Z', '+00:00'))
            age = datetime.datetime.now() - flagged_dt.replace(tzinfo=None)
            return age.days
            
        except Exception:
            return 0
    
    def submit_review(self, filename: str, region_id: Optional[str], 
                     rating: int, quality_score: float, 
                     comments: str = "", reviewer_id: str = "unknown",
                     metrics: Optional[Dict] = None) -> int:
        """Submit expert review with validation."""
        try:
            # Validate inputs
            self._validate_review_inputs(rating, quality_score, comments)
            
            # Submit to database
            review_id = self.db.submit_expert_review(
                filename, region_id, rating, quality_score, 
                comments, reviewer_id, metrics
            )
            
            # Log system event
            self.db.log_system_event(
                "review_submitted",
                {
                    "filename": filename,
                    "rating": rating,
                    "quality_score": quality_score,
                    "reviewer_id": reviewer_id,
                    "review_id": review_id
                }
            )
            
            # Update reviewer workload tracking
            self._update_reviewer_stats(reviewer_id, rating, quality_score)
            
            self.logger.info(f"Expert review submitted for {filename} by {reviewer_id} (rating: {rating})")
            
            return review_id
            
        except Exception as e:
            self.logger.error(f"Error submitting review: {e}")
            raise
    
    def _validate_review_inputs(self, rating: int, quality_score: float, comments: str):
        """Validate review inputs."""
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        if not (0.0 <= quality_score <= 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")
        
        if len(comments) > 1000:
            raise ValueError("Comments must be less than 1000 characters")
    
    def _update_reviewer_stats(self, reviewer_id: str, rating: int, quality_score: float):
        """Update reviewer statistics."""
        try:
            # This could be expanded to track reviewer performance
            self.logger.debug(f"Updated stats for reviewer {reviewer_id}")
        except Exception as e:
            self.logger.error(f"Error updating reviewer stats: {e}")
    
    def get_review_queue(self, reviewer_id: Optional[str] = None, 
                        max_items: int = 50) -> List[Dict]:
        """Get prioritized review queue."""
        try:
            if reviewer_id:
                # Get items assigned to specific reviewer
                pending = self.db.get_pending_reviews(max_items, None)
                assigned = [item for item in pending 
                           if item.get('assigned_reviewer') == reviewer_id]
                return assigned
            else:
                # Get highest priority unassigned items
                pending = self.db.get_pending_reviews(max_items, 2)  # Priority 2 and higher
                unassigned = [item for item in pending 
                             if not item.get('assigned_reviewer')]
                return sorted(unassigned, key=lambda x: (x.get('priority', 5), x.get('flagged_date', '')))
            
        except Exception as e:
            self.logger.error(f"Error getting review queue: {e}")
            return []
    
    def assign_reviewer(self, flagged_region_id: int, reviewer_id: str) -> bool:
        """Assign a reviewer to a flagged region."""
        try:
            success = self.db.assign_reviewer(flagged_region_id, reviewer_id)
            
            if success:
                # Log assignment
                self.db.log_system_event(
                    "reviewer_assigned",
                    {
                        "flagged_region_id": flagged_region_id,
                        "reviewer_id": reviewer_id
                    }
                )
                
                self.logger.info(f"Assigned reviewer {reviewer_id} to region {flagged_region_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error assigning reviewer: {e}")
            return False
    
    def auto_review_quality(self, filename: str, quality_metrics: Dict) -> Optional[int]:
        """Perform automatic quality review based on metrics."""
        try:
            composite_quality = quality_metrics.get('composite_quality', 0.0)
            
            # Determine if auto-review is appropriate
            if composite_quality >= self.auto_review_thresholds['excellent_threshold']:
                # Excellent quality - auto-approve
                review_id = self.submit_review(
                    filename=filename,
                    region_id=None,
                    rating=5,
                    quality_score=composite_quality,
                    comments="Auto-approved: Excellent quality metrics",
                    reviewer_id="system_auto_review",
                    metrics=quality_metrics
                )
                
                self.logger.info(f"Auto-approved {filename} with excellent quality ({composite_quality:.3f})")
                return review_id
                
            elif composite_quality < self.auto_review_thresholds['poor_threshold']:
                # Poor quality - auto-flag for review
                self.flag_for_review(
                    filename=filename,
                    region=(0, 0, 512, 512),  # Full file
                    flag_type=ExpertReviewFlag.LOW_QUALITY.value,
                    confidence=1.0 - composite_quality,
                    description=f"Auto-flagged: Poor quality metrics (score: {composite_quality:.3f})"
                )
                
                self.logger.warning(f"Auto-flagged {filename} for poor quality ({composite_quality:.3f})")
                return None
            
            # Medium quality - no auto action
            return None
            
        except Exception as e:
            self.logger.error(f"Error in auto quality review: {e}")
            return None
    
    def generate_review_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive review report."""
        try:
            stats = self.db.get_review_statistics()
            trends = self.db.get_quality_trends(days)
            
            report = {
                'generation_date': datetime.datetime.now().isoformat(),
                'period_days': days,
                'statistics': stats,
                'trends': trends,
                'summary': self._generate_summary(stats, trends),
                'recommendations': self._generate_recommendations(stats, trends)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating review report: {e}")
            return {'error': str(e)}
    
    def _generate_summary(self, stats: Dict, trends: Dict) -> Dict[str, str]:
        """Generate human-readable summary."""
        summary = {}
        
        try:
            total_reviews = stats.get('total_reviews', 0)
            pending_reviews = stats.get('pending_reviews', 0)
            avg_quality = stats.get('average_quality_score', 0)
            
            summary['overview'] = f"Total reviews: {total_reviews}, Pending: {pending_reviews}"
            summary['quality'] = f"Average quality score: {avg_quality:.3f}"
            
            if pending_reviews > 20:
                summary['alert'] = "High number of pending reviews - consider additional reviewers"
            elif avg_quality < 0.7:
                summary['alert'] = "Low average quality - review processing parameters"
            else:
                summary['status'] = "Review system operating normally"
            
        except Exception as e:
            summary['error'] = f"Error generating summary: {e}"
        
        return summary
    
    def _generate_recommendations(self, stats: Dict, trends: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            pending_reviews = stats.get('pending_reviews', 0)
            avg_quality = stats.get('average_quality_score', 0)
            flag_distribution = stats.get('flag_type_distribution', {})
            
            # Pending review recommendations
            if pending_reviews > 50:
                recommendations.append("Critical: Assign additional reviewers to reduce backlog")
            elif pending_reviews > 20:
                recommendations.append("Consider increasing reviewer capacity")
            
            # Quality recommendations
            if avg_quality < 0.6:
                recommendations.append("Review and adjust processing parameters for better quality")
            elif avg_quality < 0.7:
                recommendations.append("Monitor quality trends and consider parameter tuning")
            
            # Flag type recommendations
            if flag_distribution.get('standards_violation', 0) > 5:
                recommendations.append("High standards violations - review compliance procedures")
            
            if flag_distribution.get('feature_loss', 0) > 10:
                recommendations.append("Frequent feature loss - adjust feature preservation settings")
            
            if not recommendations:
                recommendations.append("System operating within normal parameters")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def get_reviewer_dashboard(self, reviewer_id: str) -> Dict[str, Any]:
        """Get personalized dashboard for reviewer."""
        try:
            workload = self.db.get_reviewer_workload(reviewer_id)
            assigned_items = self.get_review_queue(reviewer_id)
            
            dashboard = {
                'reviewer_id': reviewer_id,
                'workload': workload,
                'assigned_items': len(assigned_items),
                'high_priority_items': len([item for item in assigned_items 
                                          if item.get('priority', 5) <= 2]),
                'oldest_item_days': max([self._calculate_age_days(item.get('flagged_date', '')) 
                                       for item in assigned_items], default=0),
                'queue_items': assigned_items[:10]  # Show top 10
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error getting reviewer dashboard: {e}")
            return {'error': str(e)}
    
    def escalate_review(self, flagged_region_id: int, reason: str, 
                       escalated_by: str) -> bool:
        """Escalate a review to higher priority."""
        try:
            # Update priority in database
            success = self.db.bulk_update_priority("escalated", 1)  # Set to highest priority
            
            if success:
                # Log escalation
                self.db.log_system_event(
                    "review_escalated",
                    {
                        "flagged_region_id": flagged_region_id,
                        "reason": reason,
                        "escalated_by": escalated_by
                    }
                )
                
                self.logger.warning(f"Review {flagged_region_id} escalated by {escalated_by}: {reason}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error escalating review: {e}")
            return False
    
    def batch_approve_reviews(self, criteria: Dict, approver_id: str) -> int:
        """Batch approve reviews meeting certain criteria."""
        try:
            # Get reviews matching criteria
            pending_reviews = self.get_pending_reviews()
            
            approved_count = 0
            min_quality = criteria.get('min_quality_score', 0.8)
            max_age_days = criteria.get('max_age_days', 7)
            
            for review in pending_reviews:
                # Check if meets criteria
                age_days = self._calculate_age_days(review.get('flagged_date', ''))
                
                if (review.get('confidence', 0) >= min_quality and 
                    age_days <= max_age_days):
                    
                    # Auto-approve
                    self.submit_review(
                        filename=review.get('filename', ''),
                        region_id=str(review.get('id', '')),
                        rating=4,  # Good rating
                        quality_score=review.get('confidence', 0.8),
                        comments=f"Batch approved by {approver_id}",
                        reviewer_id=f"batch_approval_{approver_id}"
                    )
                    
                    approved_count += 1
            
            if approved_count > 0:
                self.db.log_system_event(
                    "batch_approval",
                    {
                        "approved_count": approved_count,
                        "criteria": criteria,
                        "approver_id": approver_id
                    }
                )
                
                self.logger.info(f"Batch approved {approved_count} reviews by {approver_id}")
            
            return approved_count
            
        except Exception as e:
            self.logger.error(f"Error in batch approval: {e}")
            return 0
    
    def export_review_data(self, output_dir: str, format: str = 'csv') -> str:
        """Export review data for external analysis."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'csv':
                filename = f"review_export_{timestamp}.csv"
                export_path = output_path / filename
                self.db.export_reviews_to_csv(str(export_path), include_metrics=True)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported review data to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting review data: {e}")
            raise
    
    def cleanup_old_reviews(self, days_old: int = 365):
        """Clean up old completed reviews."""
        try:
            self.db.cleanup_old_records(days_old)
            
            self.db.log_system_event(
                "cleanup_completed",
                {"days_old": days_old}
            )
            
            self.logger.info(f"Cleaned up reviews older than {days_old} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old reviews: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get expert review system health status."""
        try:
            stats = self.db.get_review_statistics()
            db_info = self.db.get_database_info()
            
            health = {
                'status': 'healthy',
                'database_size_mb': db_info.get('file_size_mb', 0),
                'total_reviews': stats.get('total_reviews', 0),
                'pending_reviews': stats.get('pending_reviews', 0),
                'average_quality': stats.get('average_quality_score', 0),
                'recent_activity': stats.get('reviews_last_7_days', 0)
            }
            
            # Determine health status
            pending = health['pending_reviews']
            if pending > 100:
                health['status'] = 'critical'
                health['issues'] = ['High pending review backlog']
            elif pending > 50:
                health['status'] = 'warning'
                health['issues'] = ['Moderate pending review backlog']
            elif health['average_quality'] < 0.6:
                health['status'] = 'warning'
                health['issues'] = ['Low average quality scores']
            
            return health
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}