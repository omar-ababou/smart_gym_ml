"""
Smart Gym ML - Main Demo Script
This script demonstrates the complete Smart Gym ML pipeline.
"""

import sys
import os
import numpy as np
from typing import Dict, List

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.exercise_recognition import ExerciseRecognizer
from ml_models.heart_rate_monitor import HeartRateMonitor
from ml_models.anomaly_detector import AnomalyDetector
from data_processing.data_generator import DataGenerator
from config.config import Config

def generate_demo_data() -> Dict:
    """Generate sample data for demonstration"""
    data_gen = DataGenerator()

    # Generate sample sensor data for squats
    sensor_data = data_gen.generate_sensor_window('squats', duration=100)

    # Generate sample heart rate data
    heart_rate_data = [115, 118, 120, 117, 119, 122, 116, 121, 118, 120]

    # User profile
    user_profile = {
        'age': 28,
        'weight': 75,
        'height': 175,
        'fitness_level': 4
    }

    return {
        'sensor_data': sensor_data,
        'heart_rate_data': heart_rate_data,
        'user_profile': user_profile
    }

def demo_exercise_recognition():
    """Demonstrate exercise recognition functionality"""
    print("🏃 Step 1: Exercise Recognition")
    print("-" * 30)

    # Initialize recognizer
    recognizer = ExerciseRecognizer()

    # Generate sample data
    demo_data = generate_demo_data()

    # Predict exercise
    result = recognizer.predict_exercise(demo_data['sensor_data'])

    if result['error'] is None:
        print(f"Detected Exercise: {result['exercise']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Top predictions:")
        for exercise, prob in result['probabilities'].items():
            print(f"  • {exercise}: {prob:.2%}")
    else:
        print(f"Error: {result['error']}")

    print()

def demo_heart_rate_monitoring():
    """Demonstrate heart rate monitoring functionality"""
    print("❤️ Step 2: Heart Rate Monitoring")
    print("-" * 30)

    # Initialize monitor
    monitor = HeartRateMonitor()

    # Generate sample data
    demo_data = generate_demo_data()

    # Predict heart rate
    result = monitor.predict_heart_rate(
        demo_data['sensor_data'],
        demo_data['user_profile']
    )

    if result['error'] is None:
        print(f"Predicted Heart Rate: {result['heart_rate']:.1f} BPM")
        print(f"Heart Rate Zone: {result['zone']['name']}")
        print(f"Zone Description: {result['zone']['description']}")
        print(f"Intensity Level: {result['zone']['intensity']}")
    else:
        print(f"Error: {result['error']}")

    print()


def demo_anomaly_detection():
    """Demonstrate anomaly detection functionality"""
    print("⚠️ Step 3: Anomaly Detection")
    print("-" * 30)

    # Initialize detector
    detector = AnomalyDetector()

    # Generate sample data
    demo_data = generate_demo_data()
    data_gen = DataGenerator()

    try:
        # Create proper feature vector manually (44 features total)
        sensor_features = data_gen.extract_features(demo_data['sensor_data'])  # 36 features

        # Heart rate features (6 features)
        hr_data = demo_data['heart_rate_data']
        hr_features = [
            np.mean(hr_data),  # mean HR
            np.std(hr_data),  # HR variability
            np.max(hr_data),  # max HR
            np.min(hr_data),  # min HR
            np.std(hr_data) / np.mean(hr_data) if np.mean(hr_data) > 0 else 0,  # HR variability ratio
            (hr_data[-1] - hr_data[0]) / len(hr_data) if len(hr_data) > 1 else 0  # HR trend
        ]

        # Duration features (2 features)
        duration_features = [
            len(demo_data['sensor_data']['accel_x']),  # window_size
            50.0  # sampling_rate
        ]

        # Combine all features (36 + 6 + 2 = 44)
        combined_features = np.concatenate([sensor_features, hr_features, duration_features])

        print(f"✅ Created feature vector with {len(combined_features)} features")

        # Use the model directly for anomaly detection
        if detector.model is not None:
            # Reshape for model input
            features_reshaped = combined_features.reshape(1, -1)

            # Get anomaly score (-1 = anomaly, 1 = normal)
            anomaly_prediction = detector.model.predict(features_reshaped)[0]
            anomaly_score = detector.model.decision_function(features_reshaped)[0]

            # Interpret results
            is_anomaly = anomaly_prediction == -1

            # Determine risk level based on score
            if anomaly_score < -0.5:
                risk_level = "high"
            elif anomaly_score < -0.2:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Display results
            print(f"Anomaly Detected: {'Yes' if is_anomaly else 'No'}")
            print(f"Risk Level: {risk_level.upper()}")
            print(f"Anomaly Score: {anomaly_score:.3f}")

            # Generate recommendations based on results
            if is_anomaly:
                print("⚠️ Warnings:")
                print("  • Unusual workout pattern detected")
                print("💡 Recommendations:")
                print("  • Consider taking a short break")
                print("  • Monitor heart rate more closely")
            else:
                print("✅ Workout pattern looks normal")
                print("💡 Recommendations:")
                print("  • Continue your current routine")
                print("  • Maintain good form")
        else:
            print("❌ Anomaly detection model not loaded")

    except Exception as e:
        print(f"❌ Anomaly detection failed: {str(e)}")
        # Fallback to simple safety check
        print("📊 Performing basic safety check instead...")
        hr_avg = np.mean(demo_data['heart_rate_data'])
        if hr_avg > 180:
            print("⚠️ Warning: Heart rate appears high")
        else:
            print("✅ Basic safety check passed")

    print()

def demo_session_analysis():
    """Demonstrate complete session analysis"""
    print("📊 Step 4: Complete Session Analysis")
    print("-" * 40)

    # Initialize all models
    recognizer = ExerciseRecognizer()
    monitor = HeartRateMonitor()
    detector = AnomalyDetector()

    # Generate session data (multiple windows)
    data_gen = DataGenerator()
    session_data = []

    # Simulate 10 windows of exercise data
    for i in range(10):
        exercise_type = 'squats' if i < 7 else 'rest'
        sensor_data = data_gen.generate_sensor_window(exercise_type)
        heart_rate = [115 + np.random.randint(-5, 6) for _ in range(5)]

        session_data.append({
            'sensor_data': sensor_data,
            'heart_rate_data': heart_rate,
            'exercise_type': exercise_type
        })

    # Analyze complete session
    exercise_results = recognizer.analyze_exercise_session([s['sensor_data'] for s in session_data])
    hr_results = monitor.monitor_session(session_data, {'age': 28, 'weight': 75})
    anomaly_results = detector.monitor_session(session_data)

    print("📈 Session Summary:")
    print(f"  Duration: {exercise_results['session_duration']} windows")
    print(f"  Dominant Exercise: {exercise_results['dominant_exercise']}")
    print(f"  Average Heart Rate: {hr_results['average_hr']:.1f} BPM")
    print(f"  Session Status: {anomaly_results['session_status']}")

    if hr_results['recommendations']:
        print("\n💡 Session Recommendations:")
        for rec in hr_results['recommendations'][:3]:  # Show top 3
            print(f"  • {rec}")

def main():
    """Main demonstration function"""
    print("🏋️ Smart Gym ML Model Demo")
    print("=" * 40)

    # Check if models exist
    models_exist = all(os.path.exists(path) for path in Config.MODEL_PATHS.values())

    if not models_exist:
        print("⚠️ Models not found! Please run 'python src/train_model.py' first.")
        return

    print("✅ All models loaded successfully!")
    print()

    # Run demonstrations
    try:
        demo_exercise_recognition()
        demo_heart_rate_monitoring()
        demo_anomaly_detection()
        demo_session_analysis()

        print("🎉 Demo completed successfully!")
        print("\n📚 Next Steps:")
        print("  1. Run 'python src/utils/firebase_exporter.py' to export models")
        print("  2. Integrate with your mobile app using the exported files")
        print("  3. Customize the models with your own training data")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please check that all models are properly trained.")

if __name__ == "__main__":
    main()