# Feature: Voice Input & Transcription (microphone or audio file)

import whisper
import speech_recognition as sr

def transcribe_from_file(audio_path):
    # Transcribe an audio file using Whisper (offline)
    model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy
    result = model.transcribe(audio_path)
    print("Transcribed text:", result["text"])
    return result["text"]

def transcribe_from_mic():
    # Transcribe speech from a live microphone (uses Google STT API)
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Transcription failed: {e}"

def save_transcription(text, output_file="transcript.txt"):
    # Save text to a file
    with open(output_file, "w") as f:
        f.write(text)
    print(f"Transcription saved to {output_file}")
