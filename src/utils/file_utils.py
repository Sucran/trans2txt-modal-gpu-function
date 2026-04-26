"""
File operation utility functions
Handles temporary file management, directory operations, and file I/O
"""

import os
import tempfile
from pathlib import Path
from typing import List, Callable, Any, Optional


def create_temp_file(
    content: bytes,
    suffix: str = ".mp3",
    prefix: str = "audio_"
) -> Path:
    """
    Create a temporary file with given content
    
    Args:
        content: File content as bytes
        suffix: File suffix (e.g., '.mp3', '.wav')
        prefix: File prefix
        
    Returns:
        Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=prefix
    )
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def create_temp_directory(prefix: str = "temp_") -> Path:
    """
    Create a temporary directory
    
    Args:
        prefix: Directory prefix
        
    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)


def cleanup_temp_file(file_path: Path) -> bool:
    """
    Clean up a temporary file
    
    Args:
        file_path: Path to temporary file
        
    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception:
        return False


def cleanup_temp_directory(dir_path: Path) -> bool:
    """
    Clean up a temporary directory
    
    Args:
        dir_path: Path to temporary directory
        
    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        if dir_path.exists() and dir_path.is_dir():
            import shutil
            shutil.rmtree(dir_path)
            return True
        return False
    except Exception:
        return False


def cleanup_temp_files(file_paths: List[Path]) -> int:
    """
    Clean up multiple temporary files
    
    Args:
        file_paths: List of file paths to clean up
        
    Returns:
        Number of files successfully cleaned up
    """
    cleaned = 0
    for file_path in file_paths:
        if cleanup_temp_file(file_path):
            cleaned += 1
    return cleaned


def ensure_directory_exists(dir_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object to directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory_exists_path(dir_path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't (Path version)
    
    Args:
        dir_path: Directory path as Path object
        
    Returns:
        Path object to directory
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_basename(file_path: str) -> str:
    """
    Get file basename (filename without path)
    
    Args:
        file_path: Full file path
        
    Returns:
        File basename
    """
    return os.path.basename(file_path)


def get_file_directory(file_path: str) -> Path:
    """
    Get file directory path
    
    Args:
        file_path: Full file path
        
    Returns:
        Directory path as Path object
    """
    return Path(file_path).parent


def read_file_bytes(file_path: str) -> bytes:
    """
    Read file as bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File content as bytes
    """
    with open(file_path, "rb") as f:
        return f.read()


def write_file_bytes(file_path: str, content: bytes) -> Path:
    """
    Write bytes to file
    
    Args:
        file_path: Path to file
        content: Content to write
        
    Returns:
        Path to written file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        f.write(content)
    
    return path


def safe_file_operation(
    operation: Callable,
    file_path: str,
    default: Any = None
) -> Any:
    """
    Safely execute a file operation with error handling
    
    Args:
        operation: File operation function to execute
        file_path: Path to file
        default: Default value to return on error
        
    Returns:
        Operation result or default value on error
    """
    try:
        return operation(file_path)
    except Exception:
        return default

