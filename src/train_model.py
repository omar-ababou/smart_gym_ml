"""
Smart Gym ML - Model Training Script
This script trains all the machine learning models for the Smart Gym application.
"""

import os
import sys
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.data_generator import DataGenerator
from config.config import Config


def create_directories():
    """Create necessary directories for saving models"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)


def train_exercise_recognition_model():
    """Train the exercise recognition model"""
    print("📊 Training Exercise Recognition Model...")

    # Generate synthetic data
    data_gen = DataGenerator()
    X, y = data_gen.generate_exercise_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   📈 Accuracy: {accuracy:.3f}")

    # Save model
    with open('models/exercise_recognition_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    print("   ✅ Exercise recognition model saved!")
    return rf_model


def train_heart_rate_model():
    """Train the heart rate monitoring model"""
    print("❤️ Training Heart Rate Model...")

    # Generate synthetic data
    data_gen = DataGenerator()
    X, y = data_gen.generate_heart_rate_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create neural network
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   📈 Mean Absolute Error: {test_mae:.2f} BPM")

    # Save model
    model.save('models/heart_rate_model.h5')

    print("   ✅ Heart rate model saved!")
    return model


def train_anomaly_detection_model():
    """Train the anomaly detection model"""
    print("⚠️ Training Anomaly Detection Model...")

    # Generate synthetic data
    data_gen = DataGenerator()
    X_normal, X_anomaly = data_gen.generate_anomaly_data()

    # Train Isolation Forest on normal data only
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_normal)

    # Test on combined data
    X_test = np.vstack([X_normal[:50], X_anomaly[:10]])
    y_test = np.hstack([np.ones(50), -np.ones(10)])  # 1 for normal, -1 for anomaly

    y_pred = iso_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   📈 Anomaly Detection Accuracy: {accuracy:.3f}")

    # Save model
    with open('models/anomaly_detection_model.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)

    print("   ✅ Anomaly detection model saved!")
    return iso_forest


def main():
    """Main training function"""
    print("🏋️ Training Smart Gym ML Models")
    print("=" * 40)

    # Create directories
    create_directories()

    # Train all models
    exercise_model = train_exercise_recognition_model()
    heart_rate_model = train_heart_rate_model()
    anomaly_model = train_anomaly_detection_model()

    print("\n🎉 All models trained successfully!")
    print(f"📁 Models saved in: {os.path.abspath('models')}")

    # Print model files
    model_files = os.listdir('models')
    print("\n📄 Trained model files:")
    for file in model_files:
        print(f"   • {file}")


if __name__ == "__main__":
    main()