"""
Parameter Set Management
Handles saving, loading, and generating parameter configurations
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import itertools


class ParameterSet:
    """
    Represents a set of parameters for the sleep data processing pipeline.
    
    Covers three parameter groups:
    - Conditioning pre-model
    - Scoring
    - Prediction (post-inference)
    """
    
    PARAM_SETS_DIR = Path("config/parameter_sets")
    
    def __init__(self, name: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize parameter set.
        
        Args:
            name: Load from saved parameter set by name
            config_dict: Create from dictionary configuration
        """
        if name:
            self.load_from_file(name)
        elif config_dict:
            self.name = None
            self.description = ""
            self.config = config_dict
            self.created_at = datetime.now().isoformat()
            self.created_by = os.getenv('USER', 'unknown')
            self.tags = []
        else:
            # Default configuration
            self.name = None
            self.description = "Default configuration"
            self.config = self._default_config()
            self.created_at = datetime.now().isoformat()
            self.created_by = os.getenv('USER', 'unknown')
            self.tags = []
    
    @staticmethod
    def _default_config() -> Dict:
        """Return default parameter configuration"""
        return {
            'conditioning': {
                'downsample_step_size': 1,  # PSG CSV row step; Zephyr uses zephyr_downsample_step in backend
                'spo2_interpolation_window': 30,  # seconds
                'audio_conditioning_steps': [
                    'savgol',
                    'normalization',
                    'standardization',
                ],
                'slope_threshold': 0.025,
                'scale_factor_high': 1.0,
                'scale_factor_low': 0.75,
            },
            'scoring': {
                'algorithm': 'flatline_detection',
                'variance_threshold': 0.005,
                'min_apnea_seconds': 10,
                'sliding_window_width': 15,  # seconds
                'sec_before_onset': 10,
                'sec_after_onset': 5,
            },
            'training': {
                'test_frac': 0.2,
                'split_strategy': 'random',  # random | by_night | stratified_by_night
                'balance_classes': True,
                'input_features': ['audio'],
            },
            'prediction': {
                'detection_algorithm': 'live_postprocessing',
                'activation_threshold': 0.7,
                'activation_seconds': 3,
                'seconds_since_last_treatment': 5.0,
                'inference_interval_sec': 1.0,
                'haptic_duration': 1000,
                'actuation_start_delay_min': 0,
                'efficacy_trailing_observation_sec': 25.0,
                'efficacy_spo2_recovery_fraction': 0.5,
                'efficacy_min_drop_to_count_prevented': 3.0,
                'efficacy_scale_gain_by_confidence': True,
            },
            'model': {
                'mode': 'none',
                'architecture': 'cnn',
                'auto_match_engine': True,
                'engine_s3_key': None,
                'epochs': 5,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 5,
                'glm_C': 1.0,
                'glm_max_iter': 1000,
                'glm_penalty': 'l2',
                'glm_solver': 'lbfgs',
                'xgb_n_estimators': 100,
                'xgb_max_depth': 6,
                'xgb_learning_rate': 0.1,
                'xgb_subsample': 0.8,
                'xgb_colsample_bytree': 0.8,
                'xgb_min_child_weight': 1.0,
            },
        }
    
    def load_from_file(self, name: str):
        """Load parameter set from JSON file"""
        filepath = self.PARAM_SETS_DIR / f"{name}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Parameter set '{name}' not found")

        print(f"[FILE READ] parameter set: {filepath.resolve()}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.name = data['name']
        self.description = data.get('description', '')
        self.config = data['parameters']
        self.created_at = data.get('created_at', '')
        self.created_by = data.get('created_by', 'unknown')
        self.tags = data.get('tags', [])
    
    def save(self, name: str, description: str = "", tags: List[str] = None):
        """
        Save parameter set to JSON file.
        
        Args:
            name: Name for the parameter set (will be used as filename)
            description: Description of the parameter set
            tags: Optional tags for categorization
        """
        self.PARAM_SETS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.name = name
        self.description = description
        if tags:
            self.tags = tags
        
        data = {
            'name': self.name,
            'description': self.description,
            'parameters': self.config,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'tags': self.tags
        }
        
        filepath = self.PARAM_SETS_DIR / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clone(self, new_name: str) -> 'ParameterSet':
        """Create a copy of this parameter set with a new name"""
        cloned = ParameterSet(config_dict=self.config.copy())
        cloned.name = new_name
        cloned.description = f"Cloned from {self.name}"
        cloned.created_at = datetime.now().isoformat()
        return cloned
    
    def to_dict(self) -> Dict:
        """Return parameter configuration as dictionary"""
        return self.config.copy()
    
    def to_json(self) -> str:
        """Return parameter configuration as JSON string"""
        return json.dumps(self.config, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ParameterSet':
        """Create parameter set from dictionary"""
        return cls(config_dict=config_dict)
    
    @classmethod
    def load(cls, name: str) -> 'ParameterSet':
        """Load parameter set by name"""
        return cls(name=name)
    
    @staticmethod
    def list_available() -> List[Dict]:
        """
        List all available parameter sets.
        
        Returns:
            List of dicts with name, description, created_at, tags
        """
        param_dir = ParameterSet.PARAM_SETS_DIR
        
        if not param_dir.exists():
            return []
        
        param_sets = []
        for filepath in param_dir.glob("*.json"):
            try:
                print(f"[FILE READ] parameter set listing: {filepath.resolve()}")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                param_sets.append({
                    'name': data['name'],
                    'description': data.get('description', ''),
                    'created_at': data.get('created_at', ''),
                    'tags': data.get('tags', [])
                })
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        return sorted(param_sets, key=lambda x: x['created_at'], reverse=True)
    
    @staticmethod
    def generate_sweep(base_params: 'ParameterSet', sweep_ranges: Dict[str, List]) -> List['ParameterSet']:
        """
        Generate parameter sets for grid search.
        
        Args:
            base_params: Base parameter configuration
            sweep_ranges: Dict mapping parameter paths to lists of values
                         e.g., {'scoring.variance_threshold': [0.003, 0.005, 0.01]}
        
        Returns:
            List of ParameterSet objects covering all combinations
        """
        # Parse parameter paths and extract keys
        param_keys = []
        param_values = []
        
        for path, values in sweep_ranges.items():
            param_keys.append(path)
            param_values.append(values)
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Create parameter sets
        param_sets = []
        for combo in combinations:
            # Start with base config
            config = base_params.to_dict()
            
            # Apply sweep values
            for path, value in zip(param_keys, combo):
                # Parse nested path (e.g., "scoring.variance_threshold")
                parts = path.split('.')
                current = config
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            
            # Create new parameter set
            param_set = ParameterSet.from_dict(config)
            param_set.description = f"Sweep: {', '.join([f'{k}={v}' for k, v in zip(param_keys, combo)])}"
            param_sets.append(param_set)
        
        return param_sets
    
    def delete(self):
        """Delete this parameter set file"""
        if not self.name:
            raise ValueError("Cannot delete parameter set without a name")
        
        filepath = self.PARAM_SETS_DIR / f"{self.name}.json"
        if filepath.exists():
            filepath.unlink()
    
    def validate(self) -> List[str]:
        """
        Validate parameter values against nominal ranges.
        
        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []
        
        # Conditioning validation
        cond = self.config['conditioning']
        if not (5 <= cond['spo2_interpolation_window'] <= 30):
            warnings.append(f"SpO2 window {cond['spo2_interpolation_window']} outside nominal range [5-30] sec")
        
        # Scoring validation
        score = self.config['scoring']
        if not (0 <= score['variance_threshold'] <= 0.1):
            warnings.append(f"Variance threshold {score['variance_threshold']} outside nominal range [0-0.1]")
        if not (5 <= score['sliding_window_width'] <= 30):
            warnings.append(f"Window width {score['sliding_window_width']} outside nominal range [5-30] sec")
        
        # Prediction validation (live post-processing)
        pred = self.config['prediction']
        if not (0.5 <= pred.get('activation_threshold', 0.7) <= 0.95):
            warnings.append(
                f"Activation threshold {pred.get('activation_threshold')} outside nominal range [0.5-0.95]"
            )
        cooldown = pred.get('seconds_since_last_treatment', pred.get('refractory_time', 5))
        if not (0 <= float(cooldown) <= 30):
            warnings.append(f"Cooldown {cooldown} outside nominal range [0-30] sec")
        
        return warnings
