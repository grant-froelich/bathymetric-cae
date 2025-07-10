"""
File Handling Utilities Module

This module provides utilities for file operations, validation,
and management in the bathymetric CAE pipeline.

Author: Bathymetric CAE Team
License: MIT
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import hashlib
import shutil


def get_valid_files(
    input_folder: Union[str, Path], 
    supported_formats: List[str]
) -> List[Path]:
    """
    Get list of valid files in input folder.
    
    Args:
        input_folder: Path to input folder
        supported_formats: List of supported file extensions
        
    Returns:
        List[Path]: List of valid file paths
    """
    input_path = Path(input_folder)
    valid_files = []
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")
    
    # Find files with supported extensions
    for ext in supported_formats:
        pattern = f"*{ext.lower()}"
        files = list(input_path.glob(pattern))
        valid_files.extend(files)
        
        # Also check uppercase extensions
        pattern_upper = f"*{ext.upper()}"
        files_upper = list(input_path.glob(pattern_upper))
        valid_files.extend(files_upper)
    
    # Remove duplicates and sort
    valid_files = sorted(list(set(valid_files)))
    
    logging.info(f"Found {len(valid_files)} valid files in {input_folder}")
    return valid_files


def validate_paths(*paths: Union[str, Path]) -> Dict[str, bool]:
    """
    Validate multiple paths.
    
    Args:
        *paths: Variable number of paths to validate
        
    Returns:
        Dict[str, bool]: Validation results for each path
    """
    results = {}
    
    for i, path in enumerate(paths):
        path = Path(path)
        path_key = f"path_{i}" if len(paths) > 1 else "path"
        
        results[path_key] = {
            'exists': path.exists(),
            'is_file': path.is_file() if path.exists() else False,
            'is_dir': path.is_dir() if path.exists() else False,
            'readable': os.access(path, os.R_OK) if path.exists() else False,
            'writable': os.access(path, os.W_OK) if path.exists() else False,
            'path_str': str(path)
        }
    
    return results


def ensure_directory(path: Union[str, Path], create_parents: bool = True) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        create_parents: Whether to create parent directories
        
    Returns:
        Path: Validated directory path
        
    Raises:
        PermissionError: If directory cannot be created
    """
    path = Path(path)
    
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
        return path
    
    try:
        path.mkdir(parents=create_parents, exist_ok=True)
        logging.debug(f"Created directory: {path}")
        return path
    except PermissionError as e:
        raise PermissionError(f"Cannot create directory {path}: {e}")


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dict[str, Any]: File information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'error': 'File not found', 'path': str(file_path)}
    
    try:
        stat = file_path.stat()
        
        info = {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix.lower(),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'size_gb': stat.st_size / (1024 * 1024 * 1024),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'accessed_time': stat.st_atime,
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK)
        }
        
        # Add human-readable size
        if info['size_gb'] >= 1:
            info['size_human'] = f"{info['size_gb']:.2f} GB"
        elif info['size_mb'] >= 1:
            info['size_human'] = f"{info['size_mb']:.2f} MB"
        else:
            info['size_human'] = f"{info['size_bytes']} bytes"
        
        return info
        
    except Exception as e:
        return {'error': str(e), 'path': str(file_path)}


def calculate_file_hash(
    file_path: Union[str, Path], 
    algorithm: str = 'md5',
    chunk_size: int = 8192
) -> str:
    """
    Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        chunk_size: Size of chunks to read at a time
        
    Returns:
        str: File hash
        
    Raises:
        ValueError: If algorithm is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get hash function
    if algorithm.lower() == 'md5':
        hash_func = hashlib.md5()
    elif algorithm.lower() == 'sha1':
        hash_func = hashlib.sha1()
    elif algorithm.lower() == 'sha256':
        hash_func = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Calculate hash
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        raise RuntimeError(f"Error calculating hash for {file_path}: {e}")


def safe_copy_file(
    source: Union[str, Path], 
    destination: Union[str, Path],
    overwrite: bool = False,
    verify_copy: bool = True
) -> bool:
    """
    Safely copy file with verification.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing files
        verify_copy: Whether to verify copy integrity
        
    Returns:
        bool: True if copy was successful
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileExistsError: If destination exists and overwrite is False
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {destination}")
    
    # Ensure destination directory exists
    ensure_directory(destination.parent)
    
    try:
        # Copy file
        shutil.copy2(source, destination)
        
        # Verify copy if requested
        if verify_copy:
            source_hash = calculate_file_hash(source)
            dest_hash = calculate_file_hash(destination)
            
            if source_hash != dest_hash:
                # Remove corrupted copy
                destination.unlink()
                raise RuntimeError("Copy verification failed - file corrupted")
        
        logging.debug(f"Successfully copied {source} to {destination}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to copy {source} to {destination}: {e}")
        raise


def clean_filename(filename: str, replacement: str = '_') -> str:
    """
    Clean filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid characters
        
    Returns:
        str: Cleaned filename
    """
    # Invalid characters for most filesystems
    invalid_chars = '<>:"/\\|?*'
    
    cleaned = filename
    for char in invalid_chars:
        cleaned = cleaned.replace(char, replacement)
    
    # Remove leading/trailing whitespace and dots
    cleaned = cleaned.strip(' .')
    
    # Ensure filename is not empty
    if not cleaned:
        cleaned = "unnamed_file"
    
    return cleaned


def get_unique_filename(
    directory: Union[str, Path], 
    base_name: str, 
    extension: str = '',
    max_attempts: int = 1000
) -> Path:
    """
    Get unique filename in directory.
    
    Args:
        directory: Target directory
        base_name: Base filename
        extension: File extension (with or without dot)
        max_attempts: Maximum number of attempts to find unique name
        
    Returns:
        Path: Unique file path
        
    Raises:
        RuntimeError: If unique name cannot be found
    """
    directory = Path(directory)
    
    # Ensure extension starts with dot
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    # Clean base name
    base_name = clean_filename(base_name)
    
    # Try original name first
    candidate = directory / f"{base_name}{extension}"
    if not candidate.exists():
        return candidate
    
    # Try numbered variants
    for i in range(1, max_attempts + 1):
        candidate = directory / f"{base_name}_{i}{extension}"
        if not candidate.exists():
            return candidate
    
    raise RuntimeError(f"Could not find unique filename after {max_attempts} attempts")


def batch_rename_files(
    file_list: List[Union[str, Path]], 
    name_pattern: str,
    start_index: int = 1,
    dry_run: bool = True
) -> List[Dict[str, str]]:
    """
    Batch rename files with pattern.
    
    Args:
        file_list: List of files to rename
        name_pattern: Pattern with {index} placeholder (e.g., "file_{index:03d}")
        start_index: Starting index number
        dry_run: If True, only return planned renames without executing
        
    Returns:
        List[Dict[str, str]]: List of rename operations
    """
    operations = []
    
    for i, file_path in enumerate(file_list):
        file_path = Path(file_path)
        
        if not file_path.exists():
            operations.append({
                'original': str(file_path),
                'new': None,
                'status': 'error',
                'message': 'File not found'
            })
            continue
        
        # Generate new name
        try:
            new_name = name_pattern.format(index=start_index + i)
            new_path = file_path.parent / (new_name + file_path.suffix)
            
            operation = {
                'original': str(file_path),
                'new': str(new_path),
                'status': 'planned',
                'message': 'Ready for rename'
            }
            
            # Check if target already exists
            if new_path.exists() and new_path != file_path:
                operation['status'] = 'error'
                operation['message'] = 'Target file already exists'
            
            # Execute rename if not dry run
            elif not dry_run and operation['status'] == 'planned':
                try:
                    file_path.rename(new_path)
                    operation['status'] = 'completed'
                    operation['message'] = 'Successfully renamed'
                except Exception as e:
                    operation['status'] = 'error'
                    operation['message'] = str(e)
            
            operations.append(operation)
            
        except Exception as e:
            operations.append({
                'original': str(file_path),
                'new': None,
                'status': 'error',
                'message': f'Pattern error: {e}'
            })
    
    return operations


def organize_files_by_extension(
    source_directory: Union[str, Path],
    target_directory: Optional[Union[str, Path]] = None,
    copy_files: bool = True,
    create_subdirs: bool = True
) -> Dict[str, List[str]]:
    """
    Organize files by extension into subdirectories.
    
    Args:
        source_directory: Source directory containing files
        target_directory: Target directory (uses source if None)
        copy_files: If True, copy files; if False, move files
        create_subdirs: Whether to create subdirectories for each extension
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping extensions to file lists
    """
    source_dir = Path(source_directory)
    target_dir = Path(target_directory) if target_directory else source_dir
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    ensure_directory(target_dir)
    
    # Group files by extension
    files_by_ext = {}
    for file_path in source_dir.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower() or 'no_extension'
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(str(file_path))
    
    # Organize files
    organized = {}
    for ext, file_list in files_by_ext.items():
        if create_subdirs:
            ext_dir = target_dir / ext.lstrip('.')
            ensure_directory(ext_dir)
        else:
            ext_dir = target_dir
        
        organized[ext] = []
        
        for file_path in file_list:
            source_file = Path(file_path)
            target_file = ext_dir / source_file.name
            
            # Handle name conflicts
            if target_file.exists() and target_file != source_file:
                target_file = get_unique_filename(
                    ext_dir, source_file.stem, source_file.suffix
                )
            
            try:
                if copy_files:
                    safe_copy_file(source_file, target_file)
                else:
                    source_file.rename(target_file)
                
                organized[ext].append(str(target_file))
                
            except Exception as e:
                logging.error(f"Failed to organize {source_file}: {e}")
    
    return organized


def find_duplicate_files(
    directory: Union[str, Path],
    algorithm: str = 'md5',
    recursive: bool = True
) -> Dict[str, List[str]]:
    """
    Find duplicate files based on hash comparison.
    
    Args:
        directory: Directory to search
        algorithm: Hash algorithm to use
        recursive: Whether to search recursively
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping hashes to file lists
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    file_hashes = {}
    
    # Get file iterator
    if recursive:
        files = directory.rglob('*')
    else:
        files = directory.iterdir()
    
    # Calculate hashes for all files
    for file_path in files:
        if file_path.is_file():
            try:
                file_hash = calculate_file_hash(file_path, algorithm)
                
                if file_hash not in file_hashes:
                    file_hashes[file_hash] = []
                
                file_hashes[file_hash].append(str(file_path))
                
            except Exception as e:
                logging.warning(f"Could not hash {file_path}: {e}")
    
    # Return only duplicates (hashes with multiple files)
    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
    
    logging.info(f"Found {len(duplicates)} sets of duplicate files")
    return duplicates


def cleanup_empty_directories(
    directory: Union[str, Path],
    recursive: bool = True,
    dry_run: bool = True
) -> List[str]:
    """
    Remove empty directories.
    
    Args:
        directory: Root directory to clean
        recursive: Whether to clean recursively
        dry_run: If True, only return directories that would be removed
        
    Returns:
        List[str]: List of directories that were (or would be) removed
    """
    directory = Path(directory)
    removed_dirs = []
    
    if not directory.exists():
        return removed_dirs
    
    # Get all directories
    if recursive:
        dirs = [d for d in directory.rglob('*') if d.is_dir()]
        # Sort by depth (deepest first) to remove child directories first
        dirs.sort(key=lambda x: len(x.parts), reverse=True)
    else:
        dirs = [d for d in directory.iterdir() if d.is_dir()]
    
    for dir_path in dirs:
        try:
            # Check if directory is empty
            if not any(dir_path.iterdir()):
                if not dry_run:
                    dir_path.rmdir()
                removed_dirs.append(str(dir_path))
                logging.debug(f"{'Would remove' if dry_run else 'Removed'} empty directory: {dir_path}")
                
        except Exception as e:
            logging.warning(f"Could not remove directory {dir_path}: {e}")
    
    return removed_dirs


class FileManager:
    """
    Comprehensive file management utility class.
    
    Provides high-level file management operations for the bathymetric pipeline.
    """
    
    def __init__(self, base_directory: Union[str, Path]):
        """
        Initialize file manager.
        
        Args:
            base_directory: Base directory for operations
        """
        self.base_dir = Path(base_directory)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        ensure_directory(self.base_dir)
    
    def setup_project_structure(self, subdirs: List[str]) -> Dict[str, Path]:
        """
        Setup project directory structure.
        
        Args:
            subdirs: List of subdirectory names to create
            
        Returns:
            Dict[str, Path]: Mapping of subdirectory names to paths
        """
        created_dirs = {}
        
        for subdir in subdirs:
            dir_path = ensure_directory(self.base_dir / subdir)
            created_dirs[subdir] = dir_path
            self.logger.info(f"Created directory: {dir_path}")
        
        return created_dirs
    
    def backup_file(
        self, 
        file_path: Union[str, Path],
        backup_dir: str = 'backups'
    ) -> Path:
        """
        Create backup of file.
        
        Args:
            file_path: File to backup
            backup_dir: Backup directory name
            
        Returns:
            Path: Path to backup file
        """
        file_path = Path(file_path)
        backup_path = ensure_directory(self.base_dir / backup_dir)
        
        # Create unique backup filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_file = backup_path / backup_name
        
        safe_copy_file(file_path, backup_file, verify_copy=True)
        self.logger.info(f"Created backup: {backup_file}")
        
        return backup_file
    
    def get_disk_usage(self) -> Dict[str, float]:
        """
        Get disk usage information for base directory.
        
        Returns:
            Dict[str, float]: Disk usage information in GB
        """
        try:
            usage = shutil.disk_usage(self.base_dir)
            
            return {
                'total_gb': usage.total / (1024**3),
                'used_gb': (usage.total - usage.free) / (1024**3),
                'free_gb': usage.free / (1024**3),
                'used_percent': ((usage.total - usage.free) / usage.total) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Could not get disk usage: {e}")
            return {}
    
    def cleanup_old_files(
        self, 
        pattern: str = '*',
        days_old: int = 30,
        dry_run: bool = True
    ) -> List[str]:
        """
        Clean up old files based on age.
        
        Args:
            pattern: File pattern to match
            days_old: Files older than this many days will be cleaned
            dry_run: If True, only return files that would be cleaned
            
        Returns:
            List[str]: List of files that were (or would be) cleaned
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        old_files = []
        
        for file_path in self.base_dir.rglob(pattern):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        if not dry_run:
                            file_path.unlink()
                        old_files.append(str(file_path))
                        
                except Exception as e:
                    self.logger.warning(f"Could not process {file_path}: {e}")
        
        action = "Would clean" if dry_run else "Cleaned"
        self.logger.info(f"{action} {len(old_files)} files older than {days_old} days")
        
        return old_files


def monitor_directory(
    directory: Union[str, Path],
    callback_func: callable,
    extensions: Optional[List[str]] = None,
    poll_interval: float = 1.0
) -> None:
    """
    Monitor directory for new files (simple polling-based).
    
    Args:
        directory: Directory to monitor
        callback_func: Function to call when new files are detected
        extensions: List of extensions to monitor (None for all)
        poll_interval: Polling interval in seconds
    """
    import time
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    logger = logging.getLogger('directory_monitor')
    logger.info(f"Starting directory monitoring: {directory}")
    
    # Get initial file set
    initial_files = set()
    for file_path in directory.iterdir():
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                initial_files.add(file_path)
    
    try:
        while True:
            time.sleep(poll_interval)
            
            # Get current files
            current_files = set()
            for file_path in directory.iterdir():
                if file_path.is_file():
                    if extensions is None or file_path.suffix.lower() in extensions:
                        current_files.add(file_path)
            
            # Find new files
            new_files = current_files - initial_files
            
            if new_files:
                logger.info(f"Detected {len(new_files)} new files")
                for new_file in new_files:
                    try:
                        callback_func(new_file)
                    except Exception as e:
                        logger.error(f"Callback failed for {new_file}: {e}")
            
            initial_files = current_files
            
    except KeyboardInterrupt:
        logger.info("Directory monitoring stopped")


# Import datetime for backup functionality
import datetime
