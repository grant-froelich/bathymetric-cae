"""
Expert review system for human-in-the-loop validation.
"""

import sqlite3
from typing import Dict, List, Tuple


class ExpertReviewSystem:
    """Human-in-the-loop validation system."""
    
    def __init__(self, db_path: str = "expert_reviews.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for expert reviews."""
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
        conn.close()
    
    def flag_for_review(self, filename: str, region: Tuple[int, int, int, int], 
                   flag_type: str, confidence: float):
        """Flag a region for expert review."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
            cursor.execute('''
                INSERT INTO flagged_regions (filename, x_start, y_start, x_end, y_end, flag_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, region[0], region[1], region[2], region[3], flag_type, confidence))
        
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error flagging {filename}: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get regions pending expert review."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM flagged_regions WHERE reviewed = FALSE
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
    
    def submit_review(self, filename: str, region_id: str, rating: int, 
                     quality_score: float, comments: str = ""):
        """Submit expert review."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO expert_reviews (filename, region_id, expert_rating, quality_score, comments)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, region_id, rating, quality_score, comments))
        
        conn.commit()
        conn.close()