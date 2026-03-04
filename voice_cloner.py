# Feature: Voice Cloning using a reference audio clip

try:
    from TTS.api import TTS as TTS_Clone
    _USE_COQUI = True
except ImportError:
    _USE_COQUI = False
    import pyttsx3

def clone_and_speak(text, speaker_audio, output_file="cloned.wav"):
    if _USE_COQUI:
        # Generate speech in the style of the reference speaker
        tts = TTS_Clone(model_name="tts_models/multilingual/multi-dataset/your_tts")
        tts.tts_to_file(text=text, speaker_wav=speaker_audio, file_path=output_file)
    else:
        # Fallback: pyttsx3 (voice cloning not supported, uses default voice)
        engine = pyttsx3.init()
        engine.save_to_file(text, output_file)
        engine.runAndWait()
