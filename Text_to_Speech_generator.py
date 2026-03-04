# Feature: Text-to-Speech (offline using Coqui TTS, fallback to pyttsx3)

try:
    from TTS.api import TTS
    _USE_COQUI = True
except ImportError:
    _USE_COQUI = False
    import pyttsx3

def speak_text_offline(text, output_file="output.wav"):
    if _USE_COQUI:
        # Load an English TTS model and save output to file
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        tts.tts_to_file(text=text, file_path=output_file)
    else:
        # Fallback: pyttsx3 (pure Python, no compilation required)
        engine = pyttsx3.init()
        engine.save_to_file(text, output_file)
        engine.runAndWait()
