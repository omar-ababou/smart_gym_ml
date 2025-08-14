"""
Smart Gym ML - Exercise Recognition Module
This module handles exercise recognition using machine learning.
"""

import pickle
import numpy as np
from typing import List, Dict, Any
import os

class ExerciseRecognizer:
    def __init__(self, model_path: str = 'models/exercise_recognition_model.pkl'):
        """
        Initialize the Exercise Recognizer

        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.exercise_classes = ['rest', 'squats', 'pushups', 'jumping_jacks', 'lunges', 'planks']
        self.load_model()

    def load_model(self):
        """Load the trained exercise recognition model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Exercise recognition model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                print("Please run train_model.py first!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def preprocess_sensor_data(self, sensor_data: Dict[str, List[float]]) -> np.ndarray:
        """
        Preprocess raw sensor data for prediction

        Args:
            sensor_data: Dictionary containing accelerometer and gyroscope data

        Returns:
            Preprocessed feature vector
        """
        # Extract features from sensor data
        features = []

        # Accelerometer features
        accel_x = np.array(sensor_data.get('accel_x', [0]))
        accel_y = np.array(sensor_data.get('accel_y', [0]))
        accel_z = np.array(sensor_data.get('accel_z', [0]))

        # Gyroscope features
        gyro_x = np.array(sensor_data.get('gyro_x', [0]))
        gyro_y = np.array(sensor_data.get('gyro_y', [0]))
        gyro_z = np.array(sensor_data.get('gyro_z', [0]))

        # Statistical features for each axis
        for data in [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]:
            features.extend([
                np.mean(data),
                np.std(data),
                np.max(data),
                np.min(data),
                np.percentile(data, 25),
                np.percentile(data, 75)
            ])

        return np.array(features).reshape(1, -1)

    def predict_exercise(self, sensor_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Predict exercise type from sensor data

        Args:
            sensor_data: Raw sensor data from smartphone/wearable

        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            return {
                'exercise': 'unknown',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }

        try:
            # Preprocess data
            features = self.preprocess_sensor_data(sensor_data)

            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            # Get confidence score
            confidence = max(probabilities)

            return {
                'exercise': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    self.model.classes_[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                },
                'error': None
            }

        except Exception as e:
            return {
                'exercise': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }

    def analyze_exercise_session(self, session_data: List[Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Analyze a complete exercise session

        Args:
            session_data: List of sensor data windows

        Returns:
            Session analysis results
        """
        predictions = []

        for window in session_data:
            result = self.predict_exercise(window)
            predictions.append(result)

        # Aggregate results
        exercises = [pred['exercise'] for pred in predictions if pred['error'] is None]

        if not exercises:
            return {
                'session_summary': 'No valid predictions',
                'exercise_counts': {},
                'dominant_exercise': 'unknown',
                'session_duration': len(session_data),
                'average_confidence': 0.0
            }

        # Count exercises
        exercise_counts = {}
        for exercise in exercises:
            exercise_counts[exercise] = exercise_counts.get(exercise, 0) + 1

        # Find dominant exercise
        dominant_exercise = max(exercise_counts, key=exercise_counts.get)

        # Calculate average confidence
        confidences = [pred['confidence'] for pred in predictions if pred['error'] is None]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'session_summary': f"Detected {len(set(exercises))} different exercises",
            'exercise_counts': exercise_counts,
            'dominant_exercise': dominant_exercise,
            'session_duration': len(session_data),
            'average_confidence': float(avg_confidence),
            'predictions': predictions
        }