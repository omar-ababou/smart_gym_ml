"""
Smart Gym ML - Anomaly Detection Module
This module detects unusual patterns in exercise data that might indicate safety issues.
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import os

class AnomalyDetector:
    def __init__(self, model_path: str = 'models/anomaly_detection_model.pkl'):
        """
        Initialize the Anomaly Detector

        Args:
            model_path: Path to the trained anomaly detection model
        """
        self.model_path = model_path
        self.model = None
        self.anomaly_threshold = -0.1  # Threshold for anomaly classification
        self.load_model()

    def load_model(self):
        """Load the trained anomaly detection model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Anomaly detection model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                print("Please run train_model.py first!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def preprocess_sensor_data(self, sensor_data: Dict[str, List[float]],
                             heart_rate_data: List[float] = None) -> np.ndarray:
        """
        Preprocess sensor and physiological data for anomaly detection

        Args:
            sensor_data: Raw sensor data from smartphone/wearable
            heart_rate_data: Heart rate measurements

        Returns:
            Preprocessed feature vector
        """
        features = []

        # Movement data
        accel_x = np.array(sensor_data.get('accel_x', [0]))
        accel_y = np.array(sensor_data.get('accel_y', [0]))
        accel_z = np.array(sensor_data.get('accel_z', [0]))
        gyro_x = np.array(sensor_data.get('gyro_x', [0]))
        gyro_y = np.array(sensor_data.get('gyro_y', [0]))
        gyro_z = np.array(sensor_data.get('gyro_z', [0]))

        # Calculate movement features
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        # Movement pattern features
        features.extend([
            np.mean(accel_magnitude),
            np.std(accel_magnitude),
            np.max(accel_magnitude),
            np.mean(gyro_magnitude),
            np.std(gyro_magnitude),
            np.max(gyro_magnitude)
        ])

        # Jerk (rate of acceleration change) - indicator of sudden movements
        accel_jerk = np.diff(accel_magnitude)
        if len(accel_jerk) > 0:
            features.extend([
                np.mean(np.abs(accel_jerk)),
                np.max(np.abs(accel_jerk)),
                np.std(accel_jerk)
            ])
        else:
            features.extend([0, 0, 0])

        # Heart rate features (if available)
        if heart_rate_data and len(heart_rate_data) > 0:
            hr_array = np.array(heart_rate_data)
            features.extend([
                np.mean(hr_array),
                np.std(hr_array),
                np.max(hr_array),
                np.min(hr_array)
            ])

            # Heart rate variability
            if len(hr_array) > 1:
                hr_diff = np.diff(hr_array)
                features.extend([
                    np.mean(np.abs(hr_diff)),
                    np.std(hr_diff)
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([80, 10, 120, 70, 5, 3])  # Default HR features

        # Duration and frequency features
        features.extend([
            len(accel_magnitude),  # Duration
            sensor_data.get('sampling_rate', 50)  # Sampling frequency
        ])

        return np.array(features).reshape(1, -1)

    def detect_anomaly(self, sensor_data: Dict[str, List[float]],
                      heart_rate_data: List[float] = None,
                      exercise_type: str = 'unknown') -> Dict[str, Any]:
        """
        Detect anomalies in exercise data

        Args:
            sensor_data: Raw sensor data
            heart_rate_data: Heart rate measurements
            exercise_type: Type of exercise being performed

        Returns:
            Anomaly detection results
        """
        if self.model is None:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'risk_level': 'unknown',
                'warnings': ['Model not loaded'],
                'recommendations': []
            }

        try:
            # Preprocess data
            features = self.preprocess_sensor_data(sensor_data, heart_rate_data)

            # Get anomaly score
            anomaly_score = self.model.decision_function(features)[0]
            is_anomaly = anomaly_score < self.anomaly_threshold

            # Determine risk level and generate warnings
            risk_level, warnings, recommendations = self.analyze_risk(
                anomaly_score, sensor_data, heart_rate_data, exercise_type
            )

            return {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'risk_level': risk_level,
                'warnings': warnings,
                'recommendations': recommendations,
                'error': None
            }

        except Exception as e:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'risk_level': 'error',
                'warnings': [f"Detection error: {str(e)}"],
                'recommendations': ['Check sensor connection'],
                'error': str(e)
            }

    def analyze_risk(self, anomaly_score: float, sensor_data: Dict[str, List[float]],
                    heart_rate_data: List[float], exercise_type: str) -> Tuple[str, List[str], List[str]]:
        """
        Analyze risk level and generate warnings/recommendations

        Args:
            anomaly_score: Anomaly score from the model
            sensor_data: Raw sensor data
            heart_rate_data: Heart rate data
            exercise_type: Type of exercise

        Returns:
            Tuple of (risk_level, warnings, recommendations)
        """
        warnings = []
        recommendations = []

        # Determine risk level based on anomaly score
        if anomaly_score < -0.3:
            risk_level = 'high'
        elif anomaly_score < -0.1:
            risk_level = 'medium'
        elif anomaly_score < 0.1:
            risk_level = 'low'
        else:
            risk_level = 'normal'

        # Analyze specific patterns for warnings

        # Check for excessive movement intensity
        accel_data = np.array(sensor_data.get('accel_x', []) +
                             sensor_data.get('accel_y', []) +
                             sensor_data.get('accel_z', []))
        if len(accel_data) > 0 and np.max(np.abs(accel_data)) > 20:
            warnings.append("Excessive movement intensity detected")
            recommendations.append("Consider reducing workout intensity")

        # Check for irregular heart rate patterns
        if heart_rate_data and len(heart_rate_data) > 0:
            hr_array = np.array(heart_rate_data)

            # Dangerously high heart rate
            if np.max(hr_array) > 190:
                warnings.append("Dangerously high heart rate detected")
                recommendations.append("Stop exercise immediately and rest")
                risk_level = 'high'

            # Rapid heart rate changes
            if len(hr_array) > 1:
                hr_changes = np.diff(hr_array)
                if np.max(np.abs(hr_changes)) > 30:
                    warnings.append("Rapid heart rate fluctuations detected")
                    recommendations.append("Take a short break and monitor heart rate")

        # Exercise-specific anomaly analysis
        if exercise_type == 'squats':
            # Check for improper squat patterns
            gyro_data = np.array(sensor_data.get('gyro_y', []))
            if len(gyro_data) > 0 and np.std(gyro_data) < 0.5:
                warnings.append("Possible improper squat form detected")
                recommendations.append("Focus on proper knee and hip movement")

        elif exercise_type == 'pushups':
            # Check for pushup form issues
            accel_z = np.array(sensor_data.get('accel_z', []))
            if len(accel_z) > 0 and np.mean(accel_z) > 12:
                warnings.append("Possible pushup form issue detected")
                recommendations.append("Ensure full range of motion")

        # General recommendations based on risk level
        if risk_level == 'high':
            recommendations.insert(0, "STOP EXERCISE IMMEDIATELY - Potential safety risk detected")
        elif risk_level == 'medium':
            recommendations.insert(0, "Caution advised - Monitor form and intensity")
        elif risk_level == 'low':
            recommendations.append("Minor irregularity detected - Stay alert")

        # Add positive feedback for normal sessions
        if risk_level == 'normal' and not warnings:
            recommendations.append("Exercise pattern looks normal - keep up the good work!")

        return risk_level, warnings, recommendations

    def monitor_session(self, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor an entire exercise session for anomalies

        Args:
            session_data: List of data windows with sensor and heart rate data

        Returns:
            Session anomaly analysis results
        """
        anomaly_results = []
        high_risk_count = 0
        medium_risk_count = 0
        all_warnings = []
        all_recommendations = []

        for i, window in enumerate(session_data):
            sensor_data = window.get('sensor_data', {})
            heart_rate_data = window.get('heart_rate_data', [])
            exercise_type = window.get('exercise_type', 'unknown')

            result = self.detect_anomaly(sensor_data, heart_rate_data, exercise_type)
            result['timestamp'] = i
            anomaly_results.append(result)

            # Count risk levels
            if result['risk_level'] == 'high':
                high_risk_count += 1
            elif result['risk_level'] == 'medium':
                medium_risk_count += 1

            # Collect unique warnings and recommendations
            for warning in result['warnings']:
                if warning not in all_warnings:
                    all_warnings.append(warning)

            for rec in result['recommendations']:
                if rec not in all_recommendations:
                    all_recommendations.append(rec)

        # Overall session assessment
        if high_risk_count > 0:
            session_risk = 'high'
            session_status = 'UNSAFE - Immediate attention required'
        elif medium_risk_count > len(session_data) * 0.3:
            session_risk = 'medium'
            session_status = 'CAUTION - Monitor closely'
        elif medium_risk_count > 0:
            session_risk = 'low'
            session_status = 'MINOR ISSUES - Stay alert'
        else:
            session_risk = 'normal'
            session_status = 'SAFE - No significant anomalies detected'

        return {
            'session_status': session_status,
            'overall_risk': session_risk,
            'high_risk_incidents': high_risk_count,
            'medium_risk_incidents': medium_risk_count,
            'total_anomalies': sum(1 for r in anomaly_results if r['is_anomaly']),
            'session_warnings': all_warnings,
            'session_recommendations': all_recommendations,
            'detailed_results': anomaly_results,
            'session_summary': f"Analyzed {len(session_data)} data windows"
        }