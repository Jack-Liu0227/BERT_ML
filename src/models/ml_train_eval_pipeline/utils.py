"""
Utility functions for the training pipeline.
"""
import os
import sys
import datetime

def to_long_path(path: str) -> str:
    """Converts a path to a Windows long path by prefixing it with \\\\?\\."""
    if sys.platform == 'win32':
        abs_path = os.path.abspath(path)
        if abs_path.startswith('\\\\?\\'):
            return abs_path
        return '\\\\?\\' + abs_path
    return path

def now():
    """Return current time string for logging."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 