from typing import Literal
class Annotation:
    def __init__(
        self, 
        type: Literal['info', 'error', 'warning'],
        category: Literal['funnel', 'conical flask', 'burette', 'lab goggles', 'white tile'],
        message: str,
        start_seconds: int,
        end_seconds: int,
    ):
        assert type in ['info', 'error', 'warning'], f'Invalid annotation type: {type}'
        self.type = type
        self.category = category
        self.message = message
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        
    def __repr__(self):
        return f'Annotation(type={self.type}, category={self.category}, message={self.message}, start_seconds={self.start_seconds}, end_seconds={self.end_seconds})'
    
    def __str__(self):
        return f'{self.type}: {self.message} ({self.start_seconds}-{self.end_seconds}s)'
    
    def to_dict(self):
        return {
            'type': self.type,
            'category': self.category,
            'message': self.message,
            'start_seconds': self.start_seconds,
            'end_seconds': self.end_seconds
        }