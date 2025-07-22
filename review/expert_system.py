"""
Expert review system for human-in-the-loop validation.
"""

import sqlite3
import logging
from typing import Dict, List, Tuple


class ExpertReviewSystem:
    """Human-in-the-loop validation system."""
    
    def __init__(self, db_path: str = "expert_reviews.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for expert reviews."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expert_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    region_id TEXT,
                    expert_rating INTEGER,
                    quality_score REAL,
                    comments TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS flagged_regions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    x_start INTEGER,
                    y_start INTEGER,
                    x_end INTEGER,
                    y_end INTEGER,
                    flag_type TEXT,
                    confidence REAL,
                    reviewed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            self.logger.info("Expert review database initialized successfully")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def flag_for_review(self, filename: str, region: Tuple[int, int, int, int], 
                       flag_type: str, confidence: float):
        """Flag a region for expert review."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
            cursor.execute('''
                INSERT INTO flagged_regions (filename, x_start, y_start, x_end, y_end, flag_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, region[0], region[1], region[2], region[3], flag_type, confidence))
        
            conn.commit()
            self.logger.info(f"Flagged {filename} for expert review: {flag_type}")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error flagging {filename}: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get regions pending expert review."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM flagged_regions WHERE reviewed = FALSE
            ''')
            
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            pending_reviews = [dict(zip(columns, row)) for row in results]
            
            self.logger.info(f"Retrieved {len(pending_reviews)} pending reviews")
            return pending_reviews
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error retrieving pending reviews: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def submit_review(self, filename: str, region_id: str, rating: int, 
                     quality_score: float, comments: str = ""):
        """Submit expert review."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO expert_reviews (filename, region_id, expert_rating, quality_score, comments)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, region_id, rating, quality_score, comments))
            
            conn.commit()
            self.logger.info(f"Expert review submitted for {filename}")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error submitting review for {filename}: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def mark_review_complete(self, region_id: int):
        """Mark a flagged region as reviewed."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE flagged_regions SET reviewed = TRUE WHERE id = ?
            ''', (region_id,))
            
            conn.commit()
            self.logger.info(f"Marked region {region_id} as reviewed")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error marking region {region_id} as reviewed: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_review_statistics(self) -> Dict:
        """Get review statistics."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total flagged regions
            cursor.execute('SELECT COUNT(*) FROM flagged_regions')
            total_flagged = cursor.fetchone()[0]
            
            # Get reviewed regions
            cursor.execute('SELECT COUNT(*) FROM flagged_regions WHERE reviewed = TRUE')
            total_reviewed = cursor.fetchone()[0]
            
            # Get pending reviews
            pending_reviews = total_flagged - total_reviewed
            
            # Get review completion rate
            completion_rate = (total_reviewed / total_flagged * 100) if total_flagged > 0 else 0
            
            stats = {
                'total_flagged': total_flagged,
                'total_reviewed': total_reviewed,
                'pending_reviews': pending_reviews,
                'completion_rate': completion_rate
            }
            
            self.logger.info(f"Review statistics: {stats}")
            return stats
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error getting statistics: {e}")
            return {}
        finally:
            if conn:
                conn.close()
                
    def export_review_data(self, output_folder: str = "expert_reviews"):
        """Export review data to JSON files."""
        import json
        from pathlib import Path
        
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Export flagged regions
            cursor.execute('SELECT * FROM flagged_regions')
            flagged_columns = [desc[0] for desc in cursor.description]
            flagged_data = [dict(zip(flagged_columns, row)) for row in cursor.fetchall()]
            
            flagged_file = output_path / "flagged_regions.json"
            with open(flagged_file, 'w') as f:
                json.dump(flagged_data, f, indent=2, default=str)
            
            # Export expert reviews
            cursor.execute('SELECT * FROM expert_reviews')
            review_columns = [desc[0] for desc in cursor.description]
            review_data = [dict(zip(review_columns, row)) for row in cursor.fetchall()]
            
            review_file = output_path / "expert_reviews.json"
            with open(review_file, 'w') as f:
                json.dump(review_data, f, indent=2, default=str)
            
            # Export summary statistics
            stats = self.get_review_statistics()
            stats_file = output_path / "review_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"Review data exported to {output_folder}/")
            return True
            
        except (sqlite3.Error, IOError) as e:
            self.logger.error(f"Error exporting review data: {e}")
            return False
        finally:
            if conn:
                conn.close()