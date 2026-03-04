from flask import Flask, render_template, request, redirect, url_for
import os

from voice_transcriber import transcribe_from_file
from Text_to_Speech_generator import speak_text_offline
from voice_cloner import clone_and_speak
from voice_rag_agent import voice_query_to_answer
from emotion_detector import predict_emotion
from podcast_summarizer import summarize_podcast

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio = request.files['audio']
    path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    audio.save(path)

    result = transcribe_from_file(path)
    return render_template('result.html', title='Transcription Result', result=result)

@app.route('/tts', methods=['POST'])
def tts():
    text = request.form['text']
    speak_text_offline(text, output_file='static/tts_output.wav')
    return render_template('result.html', title='Text-to-Speech Output', audio_url=url_for('static', filename='tts_output.wav'))

@app.route('/clone', methods=['POST'])
def clone():
    text = request.form['text']
    speaker_audio = request.files['speaker_audio']
    path = os.path.join(app.config['UPLOAD_FOLDER'], speaker_audio.filename)
    speaker_audio.save(path)

    clone_and_speak(text, speaker_audio=path, output_file='static/cloned.wav')
    return render_template('result.html', title='Cloned Voice Output', audio_url=url_for('static', filename='cloned.wav'))

@app.route('/emotion', methods=['POST'])
def emotion():
    audio = request.files['audio']
    path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    audio.save(path)

    label = predict_emotion(path)
    return render_template('result.html', title='Detected Emotion', result=f'Emotion: {label}')

@app.route('/qa', methods=['POST'])
def qa():
    audio = request.files['audio']
    path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    audio.save(path)

    docs = [
        "Python is a programming language.",
        "Whisper is an ASR model.",
        "The capital of France is Paris."
    ]
    answer = voice_query_to_answer(path, documents=docs)
    return render_template('result.html', title='Answer to Your Voice Question', result=answer)

@app.route('/summarize', methods=['POST'])
def summarize():
    podcast = request.files['podcast']
    path = os.path.join(app.config['UPLOAD_FOLDER'], podcast.filename)
    podcast.save(path)

    summary = summarize_podcast(path)
    return render_template('result.html', title='Podcast Summary', result=summary)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
