from . import db  # Import the db instance from __init__.py
from datetime import datetime

class VideoDatabase(db.Model):
    __tablename__ = 'video_database'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    video_name = db.Column(db.String(255), nullable=False)
    
    # Additional fields
    duration = db.Column(db.Float)  # Video duration in seconds
    status = db.Column(db.String(50), default='pending')  # Processing status
    description = db.Column(db.Text)  # Optional text field for video description
    
    def __repr__(self):
        return f"<VideoRecord {self.video_name}, User {self.user_id}>"