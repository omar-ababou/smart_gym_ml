# Smart Gym ML Model

A machine learning model for exercise recognition and heart rate anomaly detection using IoT sensor data.

## Overview

This project implements a 3-step ML pipeline:
1. **Exercise Recognition**: Detects exercises from accelerometer/gyroscope data
2. **Heart Rate Monitoring**: Tracks BPM during exercises
3. **Anomaly Detection**: Identifies dangerous heart rate patterns

## Hardware Requirements
- MPU-6050 (Accelerometer/Gyroscope)
- MAX30102 (Heart Rate Sensor)

## Setup Instructions

### 1. Environment Setup
```bash
# Create conda environment
conda create -n smartgym python=3.9
conda activate smartgym

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Model
```bash
python src/main.py
```

### 3. Training (Optional)
```bash
python src/train_model.py
```

## Firebase Integration

The trained model is exported as `models/smartgym_model.tflite` for mobile integration.

## Project Structure
```
smart_gym_ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
├── src/
│   ├── data_processing/
│   ├── ml_models/
│   └── utils/
├── notebooks/
├── config/
└── tests/
```