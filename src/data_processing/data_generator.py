"""
Smart Gym ML - Data Generator Module
This module generates synthetic training data for all ML models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import random

class DataGenerator:
    def __init__(self, random_seed: int = 42):
        """
        Initialize the Data Generator

        Args:
            random_seed: Seed for reproducible random data generation
        """
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Exercise patterns for different activities
        self.exercise_patterns = {
            'rest': {'accel_range': (0.5, 2.0), 'gyro_range': (0.1, 0.5), 'frequency': 0.1},
            'squats': {'accel_range': (5.0, 15.0), 'gyro_range': (2.0, 8.0), 'frequency': 0.5},
            'pushups': {'accel_range': (3.0, 12.0), 'gyro_range': (1.5, 6.0), 'frequency': 0.7},
            'jumping_jacks': {'accel_range': (8.0, 20.0), 'gyro_range': (3.0, 10.0), 'frequency': 1.2},
            'lunges': {'accel_range': (4.0, 14.0), 'gyro_range': (2.5, 7.0), 'frequency': 0.4},
            'planks': {'accel_range': (0.8, 3.0), 'gyro_range': (0.5, 2.0), 'frequency': 0.2}
        }

    def generate_sensor_window(self, exercise_type: str, duration: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate synthetic sensor data for a specific exercise

        Args:
            exercise_type: Type of exercise to simulate
            duration: Number of data points to generate

        Returns:
            Dictionary containing synthetic sensor data
        """
        pattern = self.exercise_patterns.get(exercise_type, self.exercise_patterns['rest'])

        # Generate time series
        t = np.linspace(0, duration/50, duration)  # 50Hz sampling rate

        # Base movement patterns
        frequency = pattern['frequency']
        accel_range = pattern['accel_range']
        gyro_range = pattern['gyro_range']

        # Generate accelerometer data with exercise-specific patterns
        accel_x = np.random.normal(0, accel_range[1]/3, duration)
        accel_y = np.random.normal(0, accel_range[1]/3, duration)
        accel_z = 9.81 + np.random.normal(0, accel_range[1]/4, duration)  # Include gravity

        # Add periodic patterns for repetitive exercises
        if exercise_type in ['squats', 'pushups', 'jumping_jacks', 'lunges']:
            periodic_pattern = np.sin(2 * np.pi * frequency * t) * accel_range[1]/2
            accel_y += periodic_pattern

            if exercise_type == 'jumping_jacks':
                accel_x += np.cos(2 * np.pi * frequency * t) * accel_range[1]/3

        # Generate gyroscope data
        gyro_x = np.random.normal(0, gyro_range[1]/3, duration)
        gyro_y = np.random.normal(0, gyro_range[1]/3, duration)
        gyro_z = np.random.normal(0, gyro_range[1]/3, duration)

        # Add rotational patterns for specific exercises
        if exercise_type == 'squats':
            gyro_y += np.sin(2 * np.pi * frequency * t) * gyro_range[1]/2
        elif exercise_type == 'pushups':
            gyro_x += np.sin(2 * np.pi * frequency * t) * gyro_range[1]/2

        return {
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }

    def extract_features(self, sensor_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract statistical features from sensor data

        Args:
            sensor_data: Raw sensor data

        Returns:
            Feature vector
        """
        features = []

        # Extract features for each sensor axis
        for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            data = sensor_data[key]
            features.extend([
                np.mean(data),
                np.std(data),
                np.max(data),
                np.min(data),
                np.percentile(data, 25),
                np.percentile(data, 75)
            ])

        return np.array(features)

    def generate_exercise_data(self, samples_per_exercise: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for exercise recognition

        Args:
            samples_per_exercise: Number of samples to generate per exercise type

        Returns:
            Tuple of (features, labels)
        """
        print("📊 Generating training dataset...")

        all_features = []
        all_labels = []

        for exercise_type in self.exercise_patterns.keys():
            print(f"Generating {samples_per_exercise} samples for {exercise_type}...")

            for _ in range(samples_per_exercise):
                # Generate sensor data
                sensor_data = self.generate_sensor_window(exercise_type)

                # Extract features
                features = self.extract_features(sensor_data)

                # Add some noise for robustness
                features += np.random.normal(0, 0.1, features.shape)

                all_features.append(features)
                all_labels.append(exercise_type)

        X = np.array(all_features)
        y = np.array(all_labels)

        print(f"✅ Generated {len(X)} samples with {X.shape[1]} features")
        return X, y

    def generate_heart_rate_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for heart rate prediction

        Args:
            num_samples: Number of samples to generate

        Returns:
            Tuple of (features, heart_rates)
        """
        print("❤️ Generating heart rate dataset...")

        features = []
        heart_rates = []

        for _ in range(num_samples):
            # Simulate user profile
            age = np.random.randint(18, 65)
            weight = np.random.normal(70, 15)
            height = np.random.normal(170, 10)
            fitness_level = np.random.randint(1, 6)

            # Simulate exercise intensity
            movement_intensity = np.random.exponential(2.0)
            duration_minutes = np.random.randint(1, 60)

            # Environmental factors
            temperature = np.random.normal(22, 5)
            humidity = np.random.normal(45, 15)

            # Calculate base heart rate
            resting_hr = 220 - age
            exercise_multiplier = 0.5 + (movement_intensity / 10) * 0.4
            fitness_adjustment = (6 - fitness_level) * 0.05

            predicted_hr = resting_hr * (exercise_multiplier + fitness_adjustment)

            # Add environmental effects
            if temperature > 25:
                predicted_hr += (temperature - 25) * 0.5

            # Add some realistic noise
            predicted_hr += np.random.normal(0, 8)

            # Ensure realistic range
            predicted_hr = max(50, min(200, predicted_hr))

            # Create feature vector
            feature_vector = [
                movement_intensity,
                np.random.normal(movement_intensity, movement_intensity/4),  # movement_std
                movement_intensity * 1.5,  # movement_max
                duration_minutes,
                age,
                weight,
                height,
                fitness_level,
                temperature,
                humidity
            ]

            features.append(feature_vector)
            heart_rates.append(predicted_hr)

        X = np.array(features)
        y = np.array(heart_rates)

        print(f"✅ Generated {len(X)} heart rate samples")
        return X, y

    def generate_anomaly_data(self, normal_samples: int = 800, anomaly_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for anomaly detection

        Args:
            normal_samples: Number of normal exercise samples
            anomaly_samples: Number of anomalous samples

        Returns:
            Tuple of (normal_data, anomaly_data)
        """
        print("⚠️ Generating anomaly detection dataset...")

        # Generate normal exercise data
        normal_data = []
        for _ in range(normal_samples):
            # Random normal exercise
            exercise_type = random.choice(list(self.exercise_patterns.keys()))
            sensor_data = self.generate_sensor_window(exercise_type)
            features = self.extract_features(sensor_data)

            # Add normal heart rate features
            normal_hr = np.random.normal(120, 20)
            hr_features = [normal_hr, 15, max(normal_hr + 30, 150), max(normal_hr - 20, 80), 5, 3]

            # Add duration and sampling rate
            duration_features = [100, 50]

            combined_features = np.concatenate([features, hr_features, duration_features])
            normal_data.append(combined_features)

        # Generate anomalous data
        anomaly_data = []
        for _ in range(anomaly_samples):
            # Start with normal data
            exercise_type = random.choice(list(self.exercise_patterns.keys()))
            sensor_data = self.generate_sensor_window(exercise_type)
            features = self.extract_features(sensor_data)

            # Introduce anomalies
            anomaly_type = random.choice(['extreme_movement', 'dangerous_hr', 'erratic_pattern'])

            if anomaly_type == 'extreme_movement':
                # Multiply movement features by large factor
                features[:18] *= np.random.uniform(3, 8)  # First 18 features are movement
                hr_features = [np.random.normal(140, 25), 20, 180, 100, 8, 5]

            elif anomaly_type == 'dangerous_hr':
                # Dangerous heart rate patterns
                dangerous_hr = np.random.uniform(180, 220)
                hr_features = [dangerous_hr, 30, dangerous_hr + 20, dangerous_hr - 10, 15, 8]

            elif anomaly_type == 'erratic_pattern':
                # Erratic sensor patterns
                features += np.random.normal(0, features.std() * 2, features.shape)
                hr_features = [np.random.normal(160, 40), 35, 200, 80, 20, 12]

            duration_features = [100, 50]
            combined_features = np.concatenate([features, hr_features, duration_features])
            anomaly_data.append(combined_features)

        normal_data = np.array(normal_data)
        anomaly_data = np.array(anomaly_data)

        print(f"✅ Generated {len(normal_data)} normal and {len(anomaly_data)} anomaly samples")
        return normal_data, anomaly_data