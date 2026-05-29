"""
Database Management
SQLite-based storage for jobs, results, and parameter sets
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class Database:
    """SQLite database manager for testbed"""
    
    def __init__(self, db_path: str = "config/testbed.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema if tables don't exist"""
        conn = sqlite3.connect(self.db_path)
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_name TEXT,
                status TEXT,  -- 'pending', 'running', 'completed', 'failed'
                model_path TEXT,
                parameter_set_name TEXT,
                parameters_json TEXT,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                total_nights INTEGER,
                processed_nights INTEGER,
                runtime_seconds REAL,
                error_message TEXT
            );
            
            CREATE TABLE IF NOT EXISTS results (
                result_id TEXT PRIMARY KEY,
                job_id TEXT,
                night_id TEXT,
                s3_output_path TEXT,
                metrics_json TEXT,
                processing_time REAL,
                status TEXT,
                error_message TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id)
            );
            
            CREATE TABLE IF NOT EXISTS parameter_sets (
                name TEXT PRIMARY KEY,
                description TEXT,
                parameters_json TEXT,
                created_by TEXT,
                created_at TIMESTAMP,
                tags TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
            CREATE INDEX IF NOT EXISTS idx_results_job ON results(job_id);
        ''')
        conn.commit()
        conn.close()
    
    def create_job(self, job_id: str, config: Dict) -> None:
        """Create new job record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO jobs (
                job_id, job_name, status, model_path, parameter_set_name,
                parameters_json, created_at, total_nights, processed_nights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id,
            config.get('job_name', f"Job {job_id[:8]}"),
            'pending',
            config['model'],
            config['params'].name if hasattr(config['params'], 'name') else 'custom',
            json.dumps(config['params'].to_dict()),
            datetime.now().isoformat(),
            len(config['nights']),
            0
        ))
        
        conn.commit()
        conn.close()
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and optional fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build UPDATE query dynamically
        updates = ['status = ?']
        values = [status]
        
        for key, value in kwargs.items():
            if key == 'started_at' and status == 'running':
                updates.append('started_at = ?')
                values.append(datetime.now().isoformat())
            elif key == 'completed_at' and status in ['completed', 'failed']:
                updates.append('completed_at = ?')
                values.append(datetime.now().isoformat())
            elif key == 'processed_nights':
                updates.append('processed_nights = ?')
                values.append(value)
            elif key == 'error_message':
                updates.append('error_message = ?')
                values.append(value)
            elif key == 'runtime_seconds':
                updates.append('runtime_seconds = ?')
                values.append(value)
        
        query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
        values.append(job_id)
        
        cursor.execute(query, values)
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job record by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_jobs(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get all jobs, optionally filtered by status"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute(
                'SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?',
                (status, limit)
            )
        else:
            cursor.execute(
                'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?',
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def add_result(self, job_id: str, night_id: str, result_data: Dict):
        """Add result for a specific night"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result_id = f"{job_id}_{night_id}"
        
        cursor.execute('''
            INSERT INTO results (
                result_id, job_id, night_id, s3_output_path,
                metrics_json, processing_time, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id,
            job_id,
            night_id,
            result_data.get('s3_output_path', ''),
            json.dumps(result_data.get('metrics', {})),
            result_data.get('processing_time', 0),
            result_data.get('status', 'completed'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_results(self, job_id: str) -> List[Dict]:
        """Get all results for a job"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM results WHERE job_id = ?', (job_id,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            result['metrics'] = json.loads(result['metrics_json'])
            results.append(result)
        
        return results
    
    def save_parameter_set(self, name: str, params: Dict):
        """Save parameter set to database (backup of JSON file)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO parameter_sets (
                name, description, parameters_json, created_by, created_at, tags
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            name,
            params.get('description', ''),
            json.dumps(params['parameters']),
            params.get('created_by', 'unknown'),
            params.get('created_at', datetime.now().isoformat()),
            json.dumps(params.get('tags', []))
        ))
        
        conn.commit()
        conn.close()
    
    def get_completed_jobs(self, limit: int = 50) -> List[Dict]:
        """Get completed jobs for results browser"""
        return self.get_all_jobs(status='completed', limit=limit)
    
    def get_active_jobs(self) -> List[Dict]:
        """Get pending and running jobs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM jobs 
            WHERE status IN ('pending', 'running')
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
