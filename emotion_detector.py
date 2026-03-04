# Feature 5: Emotion Detection
# Objective: Analyze a speaker's voice to detect emotional states such as happy, sad, angry, or neutral.

import torch
import librosa
import numpy as np
import joblib
from train_emotion_cnn import EmotionCNN, extract_mfcc_2d

# Initialize model
model = EmotionCNN(num_classes=8)
model.load_state_dict(torch.load("emotion_cnn.pth", map_location="cpu", weights_only=True))
model.eval()

# Load label encoder
le = joblib.load("emotion_label_encoder.pkl")

def predict_emotion(audio_path):
    mfcc = extract_mfcc_2d(audio_path)
    x = torch.tensor(mfcc[np.newaxis, np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
    return le.inverse_transform([pred])[0]

# Notes:
# - Input: WAV file path
# - Output: Emotion label (e.g., happy, sad, angry, etc.)
# - MFCCs are used as input features
# - Model trained on datasets like RAVDESS or similar