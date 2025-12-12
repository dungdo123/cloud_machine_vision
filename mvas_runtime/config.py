"""
MVAS Configuration Module

Centralized configuration management for the MVAS runtime.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list[str] = ["*"]


class StorageConfig(BaseModel):
    """Storage configuration"""
    apps_dir: Path = Path("./apps")
    data_dir: Path = Path("./data")
    results_dir: Path = Path("./data/results")
    images_dir: Path = Path("./data/images")
    temp_dir: Path = Path("./temp")
    
    def ensure_dirs(self):
        """Create all storage directories if they don't exist"""
        for path in [self.apps_dir, self.data_dir, self.results_dir, 
                     self.images_dir, self.temp_dir]:
            path.mkdir(parents=True, exist_ok=True)


class InferenceConfig(BaseModel):
    """Inference engine configuration"""
    default_device: str = "auto"  # "cpu", "cuda", "auto"
    default_precision: str = "fp32"  # "fp32", "fp16", "int8"
    warmup_iterations: int = 3
    max_batch_size: int = 1
    timeout_seconds: float = 30.0


class CameraConfig(BaseModel):
    """Camera default configuration"""
    default_timeout_ms: int = 5000
    auto_reconnect: bool = True
    reconnect_delay_seconds: float = 5.0


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = None


class MVASConfig(BaseSettings):
    """
    Main MVAS configuration.
    
    Configuration can be loaded from:
    1. Environment variables (prefixed with MVAS_)
    2. .env file
    3. config.yaml file
    4. Direct instantiation
    """
    
    # Server settings
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # Storage settings
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Inference settings
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    
    # Camera settings
    camera: CameraConfig = Field(default_factory=CameraConfig)
    
    # Logging settings
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_gpu: bool = True
    enable_streaming: bool = True
    save_results: bool = True
    save_images: bool = False
    
    class Config:
        env_prefix = "MVAS_"
        env_file = ".env"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Path) -> "MVASConfig":
        """Load configuration from YAML file"""
        import yaml
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def setup(self):
        """Initialize configuration - create directories, setup logging"""
        self.storage.ensure_dirs()
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging based on settings"""
        import logging
        
        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
        )
        
        if self.logging.file:
            handler = logging.FileHandler(self.logging.file)
            handler.setFormatter(logging.Formatter(self.logging.format))
            logging.getLogger().addHandler(handler)


# Global configuration instance
_config: Optional[MVASConfig] = None


def get_config() -> MVASConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = MVASConfig()
        _config.setup()
    return _config


def set_config(config: MVASConfig):
    """Set the global configuration instance"""
    global _config
    _config = config
    _config.setup()

