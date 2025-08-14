"""
Smart Gym ML - Heart Rate Monitoring Module
This module handles heart rate prediction and monitoring using machine learning.
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional
import os

class HeartRateMonitor:
    def __init__(self, model_path: str = 'models/heart_rate_model.h5'):
        """
        Initialize the Heart Rate Monitor

        Args:
            model_path: Path to the trained TensorFlow model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained heart rate prediction model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Heart rate model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                print("Please run train_model.py first!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def preprocess_sensor_data(self, sensor_data: Dict[str, List[float]],
                             user_profile: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Preprocess sensor data for heart rate prediction

        Args:
            sensor_data: Raw sensor data from smartphone/wearable
            user_profile: Optional user profile information

        Returns:
            Preprocessed feature vector
        """
        features = []

        # Extract movement intensity features
        accel_x = np.array(sensor_data.get('accel_x', [0]))
        accel_y = np.array(sensor_data.get('accel_y', [0]))
        accel_z = np.array(sensor_data.get('accel_z', [0]))

        # Calculate movement magnitude
        movement_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Movement features
        features.extend([
            np.mean(movement_magnitude),
            np.std(movement_magnitude),
            np.max(movement_magnitude),
            len(movement_magnitude)  # Duration
        ])

        # User profile features (with defaults)
        if user_profile:
            features.extend([
                user_profile.get('age', 30),
                user_profile.get('weight', 70),
                user_profile.get('height', 170),
                user_profile.get('fitness_level', 3)  # 1-5 scale
            ])
        else:
            features.extend([30, 70, 170, 3])  # Default values

        # Environmental features
        features.extend([
            sensor_data.get('temperature', 22),  # Room temperature
            sensor_data.get('humidity', 45)      # Humidity percentage
        ])

        return np.array(features).reshape(1, -1)

    def predict_heart_rate(self, sensor_data: Dict[str, List[float]],
                          user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict current heart rate from sensor data

        Args:
            sensor_data: Raw sensor data
            user_profile: User profile information

        Returns:
            Heart rate prediction results
        """
        if self.model is None:
            return {
                'heart_rate': 0,
                'confidence': 0.0,
                'status': 'unknown',
                'error': 'Model not loaded'
            }

        try:
            # Preprocess data
            features = self.preprocess_sensor_data(sensor_data, user_profile)

            # Make prediction
            prediction = self.model.predict(features, verbose=0)[0][0]
            predicted_hr = max(40, min(220, prediction))  # Clamp to realistic range

            # Determine status based on heart rate zones
            status = self.get_heart_rate_zone(predicted_hr, user_profile)

            # Calculate confidence based on prediction stability
            confidence = min(1.0, max(0.5, 1.0 - abs(prediction - predicted_hr) / 50))

            return {
                'heart_rate': float(predicted_hr),
                'confidence': float(confidence),
                'status': status,
                'zone': self.get_heart_rate_zone_info(predicted_hr, user_profile),
                'error': None
            }

        except Exception as e:
            return {
                'heart_rate': 0,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }

    def get_heart_rate_zone(self, heart_rate: float,
                           user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Determine heart rate zone based on age and fitness level

        Args:
            heart_rate: Current heart rate
            user_profile: User profile for age calculation

        Returns:
            Heart rate zone description
        """
        age = user_profile.get('age', 30) if user_profile else 30
        max_hr = 220 - age

        hr_percentage = (heart_rate / max_hr) * 100

        if hr_percentage < 50:
            return 'resting'
        elif hr_percentage < 60:
            return 'warm_up'
        elif hr_percentage < 70:
            return 'fat_burn'
        elif hr_percentage < 80:
            return 'aerobic'
        elif hr_percentage < 90:
            return 'anaerobic'
        else:
            return 'maximum'

    def get_heart_rate_zone_info(self, heart_rate: float,
                                user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get detailed heart rate zone information

        Args:
            heart_rate: Current heart rate
            user_profile: User profile information

        Returns:
            Detailed zone information
        """
        zone = self.get_heart_rate_zone(heart_rate, user_profile)

        zone_info = {
            'resting': {
                'name': 'Resting',
                'description': 'Recovery and rest',
                'color': 'blue',
                'intensity': 'Very Low'
            },
            'warm_up': {
                'name': 'Warm Up',
                'description': 'Light activity',
                'color': 'green',
                'intensity': 'Low'
            },
            'fat_burn': {
                'name': 'Fat Burn',
                'description': 'Fat burning zone',
                'color': 'yellow',
                'intensity': 'Moderate'
            },
            'aerobic': {
                'name': 'Aerobic',
                'description': 'Cardio fitness',
                'color': 'orange',
                'intensity': 'High'
            },
            'anaerobic': {
                'name': 'Anaerobic',
                'description': 'High intensity',
                'color': 'red',
                'intensity': 'Very High'
            },
            'maximum': {
                'name': 'Maximum',
                'description': 'Peak performance',
                'color': 'darkred',
                'intensity': 'Maximum'
            }
        }

        return zone_info.get(zone, zone_info['resting'])

    def monitor_session(self, session_data: List[Dict[str, List[float]]],
                       user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor heart rate throughout an exercise session

        Args:
            session_data: List of sensor data windows
            user_profile: User profile information

        Returns:
            Session monitoring results
        """
        heart_rates = []
        zones = []

        for window in session_data:
            result = self.predict_heart_rate(window, user_profile)
            if result['error'] is None:
                heart_rates.append(result['heart_rate'])
                zones.append(result['zone']['name'])

        if not heart_rates:
            return {
                'session_summary': 'No valid heart rate data',
                'average_hr': 0,
                'max_hr': 0,
                'min_hr': 0,
                'time_in_zones': {},
                'recommendations': ['Check sensor connection']
            }

        # Calculate statistics
        avg_hr = np.mean(heart_rates)
        max_hr = np.max(heart_rates)
        min_hr = np.min(heart_rates)

        # Time in zones
        zone_counts = {}
        for zone in zones:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

        # Generate recommendations
        recommendations = self.generate_recommendations(avg_hr, zones, user_profile)

        return {
            'session_summary': f"Monitored {len(heart_rates)} readings",
            'average_hr': float(avg_hr),
            'max_hr': float(max_hr),
            'min_hr': float(min_hr),
            'time_in_zones': zone_counts,
            'heart_rate_data': heart_rates,
            'recommendations': recommendations
        }

    def generate_recommendations(self, avg_hr: float, zones: List[str],
                               user_profile: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate workout recommendations based on heart rate data

        Args:
            avg_hr: Average heart rate during session
            zones: List of heart rate zones during session
            user_profile: User profile information

        Returns:
            List of recommendations
        """
        recommendations = []

        # Zone-based recommendations
        if zones.count('Maximum') > len(zones) * 0.2:
            recommendations.append("Consider reducing intensity - you're spending too much time at maximum heart rate")

        if zones.count('Resting') > len(zones) * 0.5:
            recommendations.append("Try increasing workout intensity for better cardiovascular benefits")

        if 'Fat Burn' in zones and zones.count('Fat Burn') > len(zones) * 0.3:
            recommendations.append("Great job staying in the fat burning zone!")

        # Heart rate specific recommendations
        age = user_profile.get('age', 30) if user_profile else 30
        target_hr = (220 - age) * 0.7  # 70% of max HR

        if avg_hr < target_hr * 0.8:
            recommendations.append("Consider increasing workout intensity to reach target heart rate")
        elif avg_hr > target_hr * 1.2:
            recommendations.append("Great intensity! Make sure to stay hydrated")

        if not recommendations:
            recommendations.append("Keep up the good work! Your heart rate patterns look healthy")

        return recommendations