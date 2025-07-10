"""
Database operations for Expert Review System.

This module handles SQLite database operations for tracking expert reviews,
flagged regions, and review history.
"""

import sqlite3
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager

from ..core.enums import ExpertReviewFlag, QualityLevel


class ReviewDatabase:
    """Database interface for expert review system."""
    
    def __init__(self, db_path: str = "expert_reviews.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables."""
        try:
            with self._get_connection() as conn:
                self._create_tables(conn)
                self._create_indexes(conn)
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        cursor = conn.cursor()
        
        # Expert reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expert_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                region_id TEXT,
                expert_rating INTEGER CHECK (expert_rating BETWEEN 1 AND 5),
                quality_score REAL CHECK (quality_score BETWEEN 0 AND 1),
                comments TEXT,
                reviewer_id TEXT,
                review_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                processing_date DATETIME,
                metrics TEXT,  -- JSON string of quality metrics
                UNIQUE(filename, region_id)
            )
        ''')
        
        # Flagged regions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flagged_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                x_start INTEGER NOT NULL,
                y_start INTEGER NOT NULL,
                x_end INTEGER NOT NULL,
                y_end INTEGER NOT NULL,
                flag_type TEXT NOT NULL,
                confidence REAL CHECK (confidence BETWEEN 0 AND 1),
                priority INTEGER DEFAULT 3,
                flagged_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewed BOOLEAN DEFAULT FALSE,
                review_id INTEGER,
                description TEXT,
                FOREIGN KEY (review_id) REFERENCES expert_reviews (id)
            )
        ''')
        
        # Review queue table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flagged_region_id INTEGER NOT NULL,
                assigned_reviewer TEXT,
                priority INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                assigned_date DATETIME,
                completed_date DATETIME,
                FOREIGN KEY (flagged_region_id) REFERENCES flagged_regions (id)
            )
        ''')
        
        # Review history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                old_values TEXT,  -- JSON string
                new_values TEXT,  -- JSON string
                changed_by TEXT,
                change_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (review_id) REFERENCES expert_reviews (id)
            )
        ''')
        
        # System log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT,  -- JSON string
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                severity TEXT DEFAULT 'INFO'
            )
        ''')
        
        conn.commit()
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for performance."""
        cursor = conn.cursor()
        
        # Indexes for common queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_expert_reviews_filename ON expert_reviews(filename)",
            "CREATE INDEX IF NOT EXISTS idx_expert_reviews_date ON expert_reviews(review_date)",
            "CREATE INDEX IF NOT EXISTS idx_flagged_regions_filename ON flagged_regions(filename)",
            "CREATE INDEX IF NOT EXISTS idx_flagged_regions_reviewed ON flagged_regions(reviewed)",
            "CREATE INDEX IF NOT EXISTS idx_flagged_regions_priority ON flagged_regions(priority)",
            "CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status)",
            "CREATE INDEX IF NOT EXISTS idx_review_queue_priority ON review_queue(priority)",
            "CREATE INDEX IF NOT EXISTS idx_review_history_review_id ON review_history(review_id)",
            "CREATE INDEX IF NOT EXISTS idx_system_log_timestamp ON system_log(timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def flag_region_for_review(self, filename: str, region: Tuple[int, int, int, int], 
                              flag_type: str, confidence: float, 
                              description: str = "") -> int:
        """Flag a region for expert review."""
        try:
            # Validate flag type
            if flag_type not in [flag.value for flag in ExpertReviewFlag]:
                raise ValueError(f"Invalid flag type: {flag_type}")
            
            # Determine priority based on flag type
            priority = self._get_priority_for_flag_type(flag_type, confidence)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO flagged_regions 
                    (filename, x_start, y_start, x_end, y_end, flag_type, confidence, priority, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (filename, region[0], region[1], region[2], region[3], 
                     flag_type, confidence, priority, description))
                
                flagged_id = cursor.lastrowid
                
                # Add to review queue
                cursor.execute('''
                    INSERT INTO review_queue (flagged_region_id, priority)
                    VALUES (?, ?)
                ''', (flagged_id, priority))
                
                conn.commit()
                
                self.logger.info(f"Flagged region in {filename} for {flag_type} review (ID: {flagged_id})")
                return flagged_id
                
        except Exception as e:
            self.logger.error(f"Error flagging region for review: {e}")
            raise
    
    def _get_priority_for_flag_type(self, flag_type: str, confidence: float) -> int:
        """Get priority level based on flag type and confidence."""
        flag_enum = ExpertReviewFlag(flag_type)
        base_priority = flag_enum.priority
        
        # Adjust priority based on confidence
        if confidence > 0.8:
            priority = max(1, base_priority - 1)  # Higher confidence = higher priority
        elif confidence < 0.3:
            priority = min(5, base_priority + 1)  # Lower confidence = lower priority
        else:
            priority = base_priority
        
        return priority
    
    def get_pending_reviews(self, limit: Optional[int] = None, 
                           priority_filter: Optional[int] = None) -> List[Dict]:
        """Get regions pending expert review."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT fr.*, rq.assigned_reviewer, rq.status, rq.assigned_date
                    FROM flagged_regions fr
                    JOIN review_queue rq ON fr.id = rq.flagged_region_id
                    WHERE fr.reviewed = FALSE
                '''
                params = []
                
                if priority_filter is not None:
                    query += " AND fr.priority <= ?"
                    params.append(priority_filter)
                
                query += " ORDER BY fr.priority ASC, fr.flagged_date ASC"
                
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            self.logger.error(f"Error getting pending reviews: {e}")
            return []
    
    def submit_expert_review(self, filename: str, region_id: Optional[str], 
                           rating: int, quality_score: float, 
                           comments: str = "", reviewer_id: str = "unknown",
                           metrics: Optional[Dict] = None) -> int:
        """Submit expert review."""
        try:
            if not (1 <= rating <= 5):
                raise ValueError("Rating must be between 1 and 5")
            
            if not (0 <= quality_score <= 1):
                raise ValueError("Quality score must be between 0 and 1")
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert metrics to JSON string
                metrics_json = None
                if metrics:
                    import json
                    metrics_json = json.dumps(metrics)
                
                cursor.execute('''
                    INSERT INTO expert_reviews 
                    (filename, region_id, expert_rating, quality_score, comments, 
                     reviewer_id, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (filename, region_id, rating, quality_score, comments, 
                     reviewer_id, metrics_json))
                
                review_id = cursor.lastrowid
                
                # Mark flagged region as reviewed if region_id provided
                if region_id:
                    cursor.execute('''
                        UPDATE flagged_regions 
                        SET reviewed = TRUE, review_id = ?
                        WHERE filename = ? AND id = ?
                    ''', (review_id, filename, region_id))
                    
                    # Update review queue
                    cursor.execute('''
                        UPDATE review_queue 
                        SET status = 'completed', completed_date = CURRENT_TIMESTAMP
                        WHERE flagged_region_id = ?
                    ''', (region_id,))
                
                conn.commit()
                
                self.logger.info(f"Expert review submitted for {filename} (ID: {review_id})")
                return review_id
                
        except Exception as e:
            self.logger.error(f"Error submitting expert review: {e}")
            raise
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM expert_reviews")
                stats['total_reviews'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM flagged_regions WHERE reviewed = FALSE")
                stats['pending_reviews'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM flagged_regions WHERE reviewed = TRUE")
                stats['completed_reviews'] = cursor.fetchone()[0]
                
                # Quality distribution
                cursor.execute('''
                    SELECT expert_rating, COUNT(*) 
                    FROM expert_reviews 
                    GROUP BY expert_rating
                ''')
                rating_distribution = dict(cursor.fetchall())
                stats['rating_distribution'] = rating_distribution
                
                # Flag type distribution
                cursor.execute('''
                    SELECT flag_type, COUNT(*) 
                    FROM flagged_regions 
                    GROUP BY flag_type
                ''')
                flag_distribution = dict(cursor.fetchall())
                stats['flag_type_distribution'] = flag_distribution
                
                # Priority distribution
                cursor.execute('''
                    SELECT priority, COUNT(*) 
                    FROM flagged_regions 
                    WHERE reviewed = FALSE
                    GROUP BY priority
                ''')
                priority_distribution = dict(cursor.fetchall())
                stats['priority_distribution'] = priority_distribution
                
                # Average quality scores
                cursor.execute("SELECT AVG(quality_score) FROM expert_reviews")
                avg_quality = cursor.fetchone()[0]
                stats['average_quality_score'] = float(avg_quality) if avg_quality else 0.0
                
                # Recent activity (last 7 days)
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM expert_reviews 
                    WHERE review_date >= datetime('now', '-7 days')
                ''')
                stats['reviews_last_7_days'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting review statistics: {e}")
            return {}
    
    def assign_reviewer(self, flagged_region_id: int, reviewer_id: str) -> bool:
        """Assign a reviewer to a flagged region."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE review_queue 
                    SET assigned_reviewer = ?, assigned_date = CURRENT_TIMESTAMP,
                        status = 'assigned'
                    WHERE flagged_region_id = ? AND status = 'pending'
                ''', (reviewer_id, flagged_region_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"Assigned reviewer {reviewer_id} to region {flagged_region_id}")
                    return True
                else:
                    self.logger.warning(f"Could not assign reviewer to region {flagged_region_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error assigning reviewer: {e}")
            return False
    
    def get_reviewer_workload(self, reviewer_id: str) -> Dict[str, Any]:
        """Get workload information for a specific reviewer."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                workload = {}
                
                # Assigned reviews
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM review_queue 
                    WHERE assigned_reviewer = ? AND status = 'assigned'
                ''', (reviewer_id,))
                workload['assigned_reviews'] = cursor.fetchone()[0]
                
                # Completed reviews
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM expert_reviews 
                    WHERE reviewer_id = ?
                ''', (reviewer_id,))
                workload['completed_reviews'] = cursor.fetchone()[0]
                
                # Average rating given
                cursor.execute('''
                    SELECT AVG(expert_rating) 
                    FROM expert_reviews 
                    WHERE reviewer_id = ?
                ''', (reviewer_id,))
                avg_rating = cursor.fetchone()[0]
                workload['average_rating'] = float(avg_rating) if avg_rating else 0.0
                
                return workload
                
        except Exception as e:
            self.logger.error(f"Error getting reviewer workload: {e}")
            return {}
    
    def log_system_event(self, event_type: str, event_data: Dict, severity: str = "INFO"):
        """Log system events for audit trail."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                import json
                event_data_json = json.dumps(event_data)
                
                cursor.execute('''
                    INSERT INTO system_log (event_type, event_data, severity)
                    VALUES (?, ?, ?)
                ''', (event_type, event_data_json, severity))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging system event: {e}")
    
    def cleanup_old_records(self, days_old: int = 365):
        """Clean up old records to maintain database performance."""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old system logs
                cursor.execute('''
                    DELETE FROM system_log 
                    WHERE timestamp < ? AND severity = 'INFO'
                ''', (cutoff_date,))
                
                deleted_logs = cursor.rowcount
                
                # Clean up old review history (keep error/warning logs)
                cursor.execute('''
                    DELETE FROM review_history 
                    WHERE change_date < ?
                ''', (cutoff_date,))
                
                deleted_history = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_logs} log entries and {deleted_history} history records")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old records: {e}")
    
    def export_reviews_to_csv(self, output_path: str, include_metrics: bool = False):
        """Export reviews to CSV format."""
        try:
            import csv
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if include_metrics:
                    query = '''
                        SELECT er.*, fr.flag_type, fr.confidence, fr.priority
                        FROM expert_reviews er
                        LEFT JOIN flagged_regions fr ON er.id = fr.review_id
                        ORDER BY er.review_date DESC
                    '''
                else:
                    query = '''
                        SELECT filename, expert_rating, quality_score, comments, 
                               reviewer_id, review_date
                        FROM expert_reviews
                        ORDER BY review_date DESC
                    '''
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                if results:
                    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        # Write header
                        writer.writerow([description[0] for description in cursor.description])
                        
                        # Write data
                        for row in results:
                            writer.writerow(row)
                    
                    self.logger.info(f"Exported {len(results)} reviews to {output_path}")
                else:
                    self.logger.warning("No reviews found to export")
                    
        except Exception as e:
            self.logger.error(f"Error exporting reviews to CSV: {e}")
            raise
    
    def get_file_review_history(self, filename: str) -> List[Dict]:
        """Get complete review history for a specific file."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT er.*, fr.flag_type, fr.confidence, fr.flagged_date,
                           rh.action, rh.change_date, rh.notes
                    FROM expert_reviews er
                    LEFT JOIN flagged_regions fr ON er.id = fr.review_id
                    LEFT JOIN review_history rh ON er.id = rh.review_id
                    WHERE er.filename = ?
                    ORDER BY er.review_date DESC, rh.change_date DESC
                ''', (filename,))
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            self.logger.error(f"Error getting file review history: {e}")
            return []
    
    def bulk_update_priority(self, flag_type: str, new_priority: int) -> int:
        """Bulk update priority for all flagged regions of a specific type."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE flagged_regions 
                    SET priority = ?
                    WHERE flag_type = ? AND reviewed = FALSE
                ''', (new_priority, flag_type))
                
                updated_count = cursor.rowcount
                conn.commit()
                
                # Also update review queue
                cursor.execute('''
                    UPDATE review_queue 
                    SET priority = ?
                    WHERE flagged_region_id IN (
                        SELECT id FROM flagged_regions 
                        WHERE flag_type = ? AND reviewed = FALSE
                    )
                ''', (new_priority, flag_type))
                
                self.logger.info(f"Updated priority for {updated_count} regions of type {flag_type}")
                return updated_count
                
        except Exception as e:
            self.logger.error(f"Error bulk updating priority: {e}")
            return 0
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, List]:
        """Get quality trends over time."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT DATE(review_date) as review_day, 
                           AVG(quality_score) as avg_quality,
                           COUNT(*) as review_count
                    FROM expert_reviews
                    WHERE review_date >= datetime('now', '-{} days')
                    GROUP BY DATE(review_date)
                    ORDER BY review_day
                '''.format(days))
                
                results = cursor.fetchall()
                
                trends = {
                    'dates': [row[0] for row in results],
                    'quality_scores': [float(row[1]) for row in results],
                    'review_counts': [row[2] for row in results]
                }
                
                return trends
                
        except Exception as e:
            self.logger.error(f"Error getting quality trends: {e}")
            return {'dates': [], 'quality_scores': [], 'review_counts': []}
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        try:
            import shutil
            
            # Ensure backup directory exists
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy database file
            shutil.copy2(self.db_path, backup_file)
            
            self.logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error backing up database: {e}")
            raise
    
    def restore_database(self, backup_path: str):
        """Restore database from backup."""
        try:
            import shutil
            
            if not Path(backup_path).exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Close any existing connections (would need connection pooling for this)
            # For now, just copy over the file
            shutil.copy2(backup_path, self.db_path)
            
            # Reinitialize to ensure schema is up to date
            self._initialize_database()
            
            self.logger.info(f"Database restored from {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error restoring database: {e}")
            raise
    
    def vacuum_database(self):
        """Optimize database by running VACUUM."""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                conn.commit()
            
            self.logger.info("Database vacuum completed")
            
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {e}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                info = {}
                
                # Database file size
                info['file_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                # Table counts
                tables = ['expert_reviews', 'flagged_regions', 'review_queue', 
                         'review_history', 'system_log']
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    info[f'{table}_count'] = cursor.fetchone()[0]
                
                # Database version info
                cursor.execute("PRAGMA user_version")
                info['user_version'] = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA schema_version")
                info['schema_version'] = cursor.fetchone()[0]
                
                return info
                
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            return {}


# Database migration utilities
class DatabaseMigration:
    """Handle database schema migrations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_current_version(self) -> int:
        """Get current database schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA user_version")
                return cursor.fetchone()[0]
        except Exception:
            return 0
    
    def migrate_to_latest(self):
        """Migrate database to latest schema version."""
        current_version = self.get_current_version()
        target_version = 1  # Current latest version
        
        if current_version >= target_version:
            self.logger.info(f"Database already at latest version {current_version}")
            return
        
        self.logger.info(f"Migrating database from version {current_version} to {target_version}")
        
        # Apply migrations
        for version in range(current_version + 1, target_version + 1):
            self._apply_migration(version)
        
        self.logger.info("Database migration completed")
    
    def _apply_migration(self, version: int):
        """Apply specific migration version."""
        migration_methods = {
            1: self._migrate_to_v1,
        }
        
        if version in migration_methods:
            migration_methods[version]()
        else:
            raise ValueError(f"Unknown migration version: {version}")
    
    def _migrate_to_v1(self):
        """Migration to version 1 - add indexes and constraints."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Add any new columns or constraints here
                # This is a placeholder for future migrations
                
                # Update version
                cursor.execute("PRAGMA user_version = 1")
                conn.commit()
                
                self.logger.info("Applied migration to version 1")
                
        except Exception as e:
            self.logger.error(f"Error applying migration to v1: {e}")
            raise


# Utility functions
def create_review_database(db_path: str) -> ReviewDatabase:
    """Factory function to create a review database."""
    return ReviewDatabase(db_path)


def migrate_database(db_path: str):
    """Migrate database to latest schema version."""
    migration = DatabaseMigration(db_path)
    migration.migrate_to_latest()


def backup_review_database(db_path: str, backup_dir: str):
    """Create a timestamped backup of the review database."""
    db = ReviewDatabase(db_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"review_db_backup_{timestamp}.db"
    backup_path = Path(backup_dir) / backup_filename
    
    db.backup_database(str(backup_path))
    return str(backup_path)