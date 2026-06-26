# EmoVision
Real-Time Facial Emotion Recognition using CNN
# EmoVision 🎭
Real-Time Facial Emotion Recognition using Deep Learning

## About
EmoVision is an AI-powered system that detects human emotions in real-time using a webcam. It uses a Convolutional Neural Network (CNN) trained on 35,000+ face images.

## Emotions Detected
- 😠 Angry
- 🤢 Disgust  
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

## Model Performance
- Training Accuracy: 85%
- Validation Accuracy: 63%
- Trained on: FER2013 Dataset
- Epochs: 100

## Tech Stack
- Python 3.12
- TensorFlow 2.21
- Keras 3.14
- OpenCV
- CNN Architecture

## How to Run
1. Install Python 3.12
2. Install required packages:
pip install tensorflow-cpu keras opencv-contrib-python numpy
3. Run:
python realtime.py

## Project Structure
- `realtime.py` - Real-time webcam emotion detection
- `best_model.h5` - Trained CNN model
