"""
MVAS Folder Watcher Input Source

Monitors a folder for new images and provides them as input.
"""

import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Set, List
from queue import Queue, Empty

import cv2
import numpy as np

from .base import InputSource, InputSourceState

logger = logging.getLogger(__name__)


class FolderWatcher(InputSource):
    """
    Folder watcher input source.
    
    Monitors a directory for new image files and provides them
    as input for processing.
    
    Configuration:
    - address: Path to folder to watch
    - patterns: List of file patterns to match (default: ["*.jpg", "*.png", "*.bmp"])
    - poll_interval: Seconds between folder checks (default: 1.0)
    - delete_after_process: Delete files after processing (default: False)
    - move_after_process: Move files to this folder after processing (default: None)
    """
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        
        self.folder_path = Path(self.config.get("address", "."))
        self.poll_interval = self.config.get("poll_interval", 1.0)
        self.delete_after_process = self.config.get("delete_after_process", False)
        self.move_folder = self.config.get("move_after_process")
        
        self._processed_files: Set[str] = set()
        self._pending_files: Queue = Queue()
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watcher = threading.Event()
    
    def connect(self) -> bool:
        """Start watching the folder"""
        try:
            logger.info(f"Connecting to folder: {self.folder_path}")
            self.state = InputSourceState.CONNECTING
            
            # Validate folder exists
            if not self.folder_path.exists():
                logger.error(f"Folder does not exist: {self.folder_path}")
                self.state = InputSourceState.ERROR
                return False
            
            if not self.folder_path.is_dir():
                logger.error(f"Path is not a directory: {self.folder_path}")
                self.state = InputSourceState.ERROR
                return False
            
            # Create move folder if specified
            if self.move_folder:
                Path(self.move_folder).mkdir(parents=True, exist_ok=True)
            
            # Start watcher thread
            self._stop_watcher.clear()
            self._watcher_thread = threading.Thread(
                target=self._watch_folder,
                daemon=True
            )
            self._watcher_thread.start()
            
            self.state = InputSourceState.CONNECTED
            logger.info(f"Folder watcher connected: {self.folder_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error connecting to folder: {e}")
            self.state = InputSourceState.ERROR
            return False
    
    def disconnect(self) -> None:
        """Stop watching the folder"""
        self.stop_streaming()
        
        self._stop_watcher.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=2.0)
        
        self.state = InputSourceState.DISCONNECTED
        logger.info(f"Folder watcher {self.source_id} disconnected")
    
    def _watch_folder(self):
        """Background thread that watches for new files"""
        logger.debug(f"Starting folder watcher for: {self.folder_path}")
        
        while not self._stop_watcher.is_set():
            try:
                # Scan for new files
                for file_path in self.folder_path.iterdir():
                    if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                        continue
                    
                    file_key = str(file_path)
                    if file_key not in self._processed_files:
                        # Check if file is fully written
                        if self._is_file_ready(file_path):
                            self._pending_files.put(file_path)
                            self._processed_files.add(file_key)
                
            except Exception as e:
                logger.error(f"Error scanning folder: {e}")
            
            # Wait for next poll
            self._stop_watcher.wait(self.poll_interval)
    
    def _is_file_ready(self, file_path: Path) -> bool:
        """Check if file is fully written (not being copied)"""
        try:
            # Check if file size is stable
            size1 = file_path.stat().st_size
            time.sleep(0.1)
            size2 = file_path.stat().st_size
            return size1 == size2 and size1 > 0
        except OSError:
            return False
    
    def grab_image(self) -> Optional[np.ndarray]:
        """Get the next available image from the folder"""
        if not self.is_connected:
            return None
        
        try:
            # Wait for a file (with timeout)
            file_path = self._pending_files.get(timeout=1.0)
            
            # Read image
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning(f"Failed to read image: {file_path}")
                return None
            
            # Handle post-processing
            self._handle_processed_file(file_path)
            
            return image
            
        except Empty:
            return None
        except Exception as e:
            logger.error(f"Error grabbing image: {e}")
            return None
    
    def _handle_processed_file(self, file_path: Path):
        """Handle file after processing"""
        try:
            if self.delete_after_process:
                file_path.unlink()
                logger.debug(f"Deleted processed file: {file_path}")
            elif self.move_folder:
                dest = Path(self.move_folder) / file_path.name
                file_path.rename(dest)
                logger.debug(f"Moved processed file to: {dest}")
        except Exception as e:
            logger.error(f"Error handling processed file: {e}")
    
    def get_pending_count(self) -> int:
        """Get number of pending files"""
        return self._pending_files.qsize()
    
    def get_images_list(self) -> List[Path]:
        """Get list of all images in folder (without processing)"""
        images = []
        for file_path in self.folder_path.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                images.append(file_path)
        return sorted(images, key=lambda p: p.stat().st_mtime)
    
    def process_single(self, file_path: str | Path) -> Optional[np.ndarray]:
        """
        Process a specific file from the folder.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image as numpy array, or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        image = cv2.imread(str(file_path))
        if image is None:
            logger.error(f"Failed to read image: {file_path}")
            return None
        
        # Mark as processed
        self._processed_files.add(str(file_path))
        self._handle_processed_file(file_path)
        
        return image
    
    def reset_processed(self):
        """Reset the processed files set (re-process all files)"""
        self._processed_files.clear()
        logger.info("Reset processed files set")
    
    def get_info(self) -> Dict[str, Any]:
        """Get folder watcher information"""
        return {
            "source_id": self.source_id,
            "type": "folder",
            "folder_path": str(self.folder_path),
            "state": self.state.value,
            "pending_files": self.get_pending_count(),
            "processed_files": len(self._processed_files),
            "poll_interval": self.poll_interval,
            "delete_after_process": self.delete_after_process,
            "move_folder": self.move_folder,
        }

