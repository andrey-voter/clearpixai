"""Configuration management for training."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration loaded from YAML file.
    
    Provides structured access to all training parameters with validation.
    """
    
    def __init__(self, config_path: str | Path):
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
        logger.info("Configuration loaded successfully")
    
    def _validate_config(self):
        """Validate that required configuration keys are present."""
        required_sections = ['data', 'model', 'training', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate critical paths exist if specified
        data_dir = Path(self.get('data.data_dir'))
        if not data_dir.exists():
            logger.warning(f"Data directory does not exist: {data_dir}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('training.learning_rate')
            0.0001
            >>> config.get('data.batch_size', 16)
            8
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Updated config: {key} = {value}")
    
    def save(self, output_path: str | Path):
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TrainingConfig(config_path={self.config_path})"


def load_config(config_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """Load training configuration with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of configuration overrides using dot notation
            Example: {'training.learning_rate': 0.001, 'data.batch_size': 16}
    
    Returns:
        Loaded configuration object
    
    Example:
        >>> config = load_config('config.yaml', {'training.learning_rate': 0.001})
    """
    config = TrainingConfig(config_path)
    
    if overrides:
        logger.info(f"Applying {len(overrides)} configuration overrides")
        for key, value in overrides.items():
            config.update(key, value)
            logger.info(f"  Override: {key} = {value}")
    
    return config

