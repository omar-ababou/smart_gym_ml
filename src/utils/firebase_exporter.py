"""
Firebase Exporter for Smart Gym ML Models - Updated to find models
"""

import os
import json
import pickle
import tensorflow as tf
from pathlib import Path
import numpy as np
import glob

class MobileModelExporter:
    def __init__(self, output_dir="mobile_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def find_models(self):
        """Automatically find all trained models in the project"""
        print("🔍 Searching for trained models...")

        # Search patterns
        search_patterns = [
            "**/*.pkl",
            "**/*.h5",
            "**/*.joblib",
            "**/*model*",
            "**/checkpoints/**",
            "**/outputs/**"
        ]

        found_files = {}

        for pattern in search_patterns:
            files = list(Path(".").glob(pattern))
            for file in files:
                if file.is_file():
                    filename = file.name.lower()
                    if "exercise" in filename and "recognition" in filename:
                        found_files['exercise'] = file
                    elif "heart" in filename and "rate" in filename:
                        found_files['heart_rate'] = file
                    elif "anomaly" in filename:
                        found_files['anomaly'] = file
                    elif "model" in filename and filename.endswith(('.pkl', '.h5', '.joblib')):
                        # Generic model file
                        found_files[filename] = file

        # Print what we found
        print("\n📁 Found model files:")
        if found_files:
            for key, path in found_files.items():
                print(f"   • {key}: {path}")
        else:
            print("   ❌ No model files found!")
            print("\n🔍 Let's check all directories:")
            self.list_all_files()

        return found_files

    def list_all_files(self):
        """List all files to help debug"""
        print("\n📂 All files in project:")
        for root, dirs, files in os.walk("."):
            if any(exclude in root for exclude in ['.git', '__pycache__', '.venv', 'node_modules']):
                continue
            for file in files:
                if file.endswith(('.pkl', '.h5', '.joblib', '.model')):
                    print(f"   • {os.path.join(root, file)}")

    def export_all_models(self):
        """Export all trained models for mobile use"""
        print("🚀 Exporting Smart Gym ML Models for Mobile...")
        print("=" * 50)

        # Find models first
        found_models = self.find_models()

        if not found_models:
            print("\n❌ No models found! Make sure you've trained your models first.")
            print("\n💡 To train models, run:")
            print("   python src/training/train_exercise_recognition.py")
            print("   python src/training/train_heart_rate_model.py")
            print("   python src/training/train_anomaly_detection.py")
            return

        # Export found models
        success_count = 0

        for model_type, model_path in found_models.items():
            try:
                if self.export_model(model_type, model_path):
                    success_count += 1
            except Exception as e:
                print(f"   ❌ Error exporting {model_type}: {e}")

        # Create configuration files
        self.create_config_files()

        print(f"\n✅ Exported {success_count} models successfully!")
        print(f"📁 Check the '{self.output_dir}' folder for mobile-ready files")

    def export_model(self, model_type, model_path):
        """Export a single model based on its type and format"""
        print(f"\n🔄 Exporting {model_type} from {model_path}")

        try:
            if str(model_path).endswith('.h5'):
                # TensorFlow/Keras model
                model = tf.keras.models.load_model(model_path)

                # Convert to TensorFlow Lite
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()

                # Save TFLite model
                output_name = f"{model_type}_model.tflite"
                tflite_path = self.output_dir / output_name
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)

                print(f"   ✅ Exported to TensorFlow Lite: {tflite_path}")
                return True

            elif str(model_path).endswith(('.pkl', '.joblib')):
                # Scikit-learn or other pickle model
                import shutil
                output_name = f"{model_type}_model.pkl"
                output_path = self.output_dir / output_name
                shutil.copy2(model_path, output_path)

                print(f"   ✅ Copied pickle model: {output_path}")
                return True

            else:
                print(f"   ⚠️  Unknown model format: {model_path}")
                return False

        except Exception as e:
            print(f"   ❌ Export failed: {e}")
            return False

    def create_config_files(self):
        """Create configuration files for mobile integration"""
        print("\n📝 Creating Configuration Files...")

        # Model specifications
        model_config = {
            "exercise_recognition": {
                "input_shape": [1, 600],
                "output_shape": [1, 6],
                "exercises": ["squats", "pushups", "jumping_jacks", "lunges", "planks", "rest"],
                "input_format": "6 sensors × 100 readings (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)",
                "sampling_rate": "50Hz recommended",
                "window_size": "2-3 seconds"
            },
            "heart_rate_prediction": {
                "input_features": ["age", "weight", "height", "fitness_level", "exercise_intensity", "movement_variability"],
                "input_shape": [1, 6],
                "output_shape": [1, 1],
                "output_range": "60-200 BPM"
            },
            "anomaly_detection": {
                "input_features": 44,
                "output": "binary (0=normal, 1=anomaly)",
                "threshold": 0.5
            }
        }

        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        print(f"   ✅ Model configuration saved: {config_path}")

        # Integration guide for Kotlin
        integration_guide = {
            "android_integration": {
                "dependencies": [
                    "implementation 'org.tensorflow:tensorflow-lite:2.8.0'",
                    "implementation 'org.tensorflow:tensorflow-lite-support:0.4.2'"
                ],
                "steps": [
                    "1. Copy .tflite files to app/src/main/assets/ folder",
                    "2. Load models using TensorFlow Lite Interpreter",
                    "3. Collect sensor data at 50Hz for 2-3 seconds",
                    "4. Format data as Float32Array[600] for exercise recognition",
                    "5. Run inference and parse results"
                ]
            },
            "kotlin_example": {
                "sensor_collection": "Use SensorManager for ACCELEROMETER and GYROSCOPE",
                "data_format": "Concatenate 6 sensor arrays: [accel_x(100), accel_y(100), accel_z(100), gyro_x(100), gyro_y(100), gyro_z(100)]",
                "inference": "Use TensorFlow Lite Interpreter.run(input, output)"
            }
        }

        guide_path = self.output_dir / "kotlin_integration_guide.json"
        with open(guide_path, 'w') as f:
            json.dump(integration_guide, f, indent=2)

        print(f"   ✅ Kotlin integration guide saved: {guide_path}")

def main():
    """Main export function"""
    exporter = MobileModelExporter()
    exporter.export_all_models()

    print("\n📦 Mobile Export Process Complete!")

if __name__ == "__main__":
    main()