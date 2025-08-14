"""
Smart Gym ML - Configuration Module
This module contains all configuration settings for the Smart Gym ML application.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class containing all settings for Smart Gym ML"""

    # Model Configuration
    MODEL_PATHS = {
        'exercise_recognition': 'models/exercise_recognition_model.pkl',
        'heart_rate_monitor': 'models/heart_rate_model.h5',
        'anomaly_detection': 'models/anomaly_detection_model.pkl'
    }

    # Data Configuration
    DATA_CONFIG = {
        'sampling_rate': 50,  # Hz
        'window_size': 100,   # Number of samples per window
        'overlap': 0.5,       # Overlap between windows
        'sensor_axes': ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    }

    # Exercise Recognition Configuration
    EXERCISE_CONFIG = {
        'classes': ['rest', 'squats', 'pushups', 'jumping_jacks', 'lunges', 'planks'],
        'confidence_threshold': 0.7,
        'min_exercise_duration': 5,  # seconds
        'max_rest_duration': 30      # seconds
    }

    # Heart Rate Configuration
    HEART_RATE_CONFIG = {
        'min_heart_rate': 40,
        'max_heart_rate': 220,
        'resting_hr_range': (60, 100),
        'danger_threshold': 190,
        'zones': {
            'resting': (0, 50),
            'warm_up': (50, 60),
            'fat_burn': (60, 70),
            'aerobic': (70, 80),
            'anaerobic': (80, 90),
            'maximum': (90, 100)
        }
    }

    # Anomaly Detection Configuration
    ANOMALY_CONFIG = {
        'contamination': 0.1,
        'anomaly_threshold': -0.1,
        'risk_levels': {
            'normal': {'threshold': 0.1, 'color': 'green'},
            'low': {'threshold': -0.1, 'color': 'yellow'},
            'medium': {'threshold': -0.3, 'color': 'orange'},
            'high': {'threshold': float('-inf'), 'color': 'red'}
        },
        'alert_conditions': {
            'excessive_movement': 20.0,
            'dangerous_heart_rate': 190,
            'rapid_hr_change': 30
        }
    }

    # Training Configuration
    TRAINING_CONFIG = {
        'exercise_recognition': {
            'n_estimators': 100,
            'random_state': 42,
            'test_size': 0.2,
            'samples_per_class': 200
        },
        'heart_rate_monitor': {
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'architecture': [64, 32, 16, 1],
            'dropout_rate': 0.2
        },
        'anomaly_detection': {
            'contamination': 0.1,
            'random_state': 42,
            'normal_samples': 800,
            'anomaly_samples': 200
        }
    }

    # Firebase Configuration
    FIREBASE_CONFIG = {
        'project_id': 'smart-gym-ml',
        'storage_bucket': 'smart-gym-models',
        'model_collection': 'ml_models',
        'config_collection': 'model_configs',
        'export_format': 'tflite',
        'model_version': '1.0.0'
    }

    # API Configuration
    API_CONFIG = {
        'max_request_size': 10 * 1024 * 1024,  # 10MB
        'rate_limit': 1000,  # requests per hour
        'timeout': 30,       # seconds
        'supported_formats': ['json', 'protobuf']
    }

    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/smart_gym_ml.log',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }

    # Performance Configuration
    PERFORMANCE_CONFIG = {
        'batch_processing': True,
        'max_batch_size': 32,
        'use_gpu': False,  # Set to True if GPU is available
        'num_threads': 4,
        'memory_limit': 2 * 1024 * 1024 * 1024  # 2GB
    }

    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """
        Get the file path for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Full path to the model file
        """
        return cls.MODEL_PATHS.get(model_name, '')

    @classmethod
    def get_exercise_classes(cls) -> list:
        """
        Get list of supported exercise classes

        Returns:
            List of exercise class names
        """
        return cls.EXERCISE_CONFIG['classes'].copy()

    @classmethod
    def get_heart_rate_zones(cls, age: int = 30) -> Dict[str, tuple]:
        """
        Get heart rate zones based on age

        Args:
            age: User's age for calculating max heart rate

        Returns:
            Dictionary of heart rate zones with BPM ranges
        """
        max_hr = 220 - age
        zones = {}

        for zone_name, (min_pct, max_pct) in cls.HEART_RATE_CONFIG['zones'].items():
            min_hr = int(max_hr * min_pct / 100)
            max_hr_zone = int(max_hr * max_pct / 100)
            zones[zone_name] = (min_hr, max_hr_zone)

        return zones

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check if model directories exist
        model_dir = os.path.dirname(list(cls.MODEL_PATHS.values())[0])
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except Exception:
                return False

        # Validate heart rate zones
        zones = cls.HEART_RATE_CONFIG['zones']
        previous_max = 0
        for zone_name, (min_pct, max_pct) in zones.items():
            if min_pct != previous_max or min_pct >= max_pct:
                return False
            previous_max = max_pct

        # Validate training parameters
        if cls.TRAINING_CONFIG['exercise_recognition']['test_size'] >= 1.0:
            return False

        return True

    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration

        Returns:
            Dictionary containing configuration summary
        """
        return {
            'exercise_classes': len(cls.EXERCISE_CONFIG['classes']),
            'heart_rate_zones': len(cls.HEART_RATE_CONFIG['zones']),
            'models_configured': len(cls.MODEL_PATHS),
            'sampling_rate': cls.DATA_CONFIG['sampling_rate'],
            'window_size': cls.DATA_CONFIG['window_size'],
            'firebase_enabled': bool(cls.FIREBASE_CONFIG['project_id']),
            'gpu_enabled': cls.PERFORMANCE_CONFIG['use_gpu']
        }