from flask import Flask, render_template, request, redirect, url_for
import os

from voice_transcriber import transcribe_from_file
from Text_to_Speech_generator import speak_text_offline
from voice_cloner import clone_and_speak
from voice_rag_agent import voice_query_to_answer
from emotion_detector import predict_emotion
from podcast_summarizer import summarize_podcast

app = Flask(__name__)
app.secret_key = 'ai-voice-assistant-secret'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}

def is_audio_file(filename):
    return os.path.splitext(filename.lower())[1] in AUDIO_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio = request.files['audio']
    if not is_audio_file(audio.filename):
        return render_template('result.html', title='Error',
            result='Please upload an audio file (.wav, .mp3, .ogg, etc.), not a PDF or other format.')
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
    if not is_audio_file(audio.filename):
        return render_template('result.html', title='Error',
            result='Please upload an audio file (.wav, .mp3, .ogg, etc.), not a PDF or other format.')
    path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    audio.save(path)

    label = predict_emotion(path)
    return render_template('result.html', title='Detected Emotion', result=f'Emotion: {label}')

@app.route('/qa', methods=['POST'])
def qa():
    audio = request.files['audio']
    if not is_audio_file(audio.filename):
        return render_template('result.html', title='Error',
            result='Voice Q&A: Please upload an AUDIO file (.wav/.mp3) as your voice question. To provide a document, use the Document field or paste text below.')
    path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    audio.save(path)

    # Build document corpus from uploaded file or pasted text
    docs = []

    # Option 1: pasted text
    doc_text = request.form.get('doc_text', '').strip()
    if doc_text:
        # Split into ~500-char chunks
        for i in range(0, len(doc_text), 500):
            chunk = doc_text[i:i+500].strip()
            if chunk:
                docs.append(chunk)

    # Option 2: uploaded document file (PDF or txt)
    doc_file = request.files.get('document')
    if doc_file and doc_file.filename:
        doc_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_file.filename)
        doc_file.save(doc_path)
        ext = os.path.splitext(doc_file.filename.lower())[1]
        if ext == '.txt':
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for i in range(0, len(content), 500):
                chunk = content[i:i+500].strip()
                if chunk:
                    docs.append(chunk)
        elif ext == '.pdf':
            try:
                import pypdf
                reader = pypdf.PdfReader(doc_path)
                content = ' '.join(page.extract_text() or '' for page in reader.pages)
                for i in range(0, len(content), 500):
                    chunk = content[i:i+500].strip()
                    if chunk:
                        docs.append(chunk)
            except ImportError:
                docs.append('PDF parsing requires pypdf. Please paste document text instead.')

    # Fallback default documents
    if not docs:
        docs = [
            "Python is a programming language.",
            "Whisper is an ASR model by OpenAI.",
            "The capital of France is Paris."
        ]

    answer = voice_query_to_answer(path, documents=docs)
    return render_template('result.html', title='Answer to Your Voice Question', result=answer)

@app.route('/summarize', methods=['POST'])
def summarize():
    podcast = request.files['podcast']
    if not is_audio_file(podcast.filename):
        return render_template('result.html', title='Error',
            result='Podcast Summarizer requires an AUDIO file (.wav, .mp3, etc.). You uploaded a non-audio file. Please upload the podcast audio, not a PDF.')
    path = os.path.join(app.config['UPLOAD_FOLDER'], podcast.filename)
    podcast.save(path)

    summary = summarize_podcast(path)
    return render_template('result.html', title='Podcast Summary', result=summary)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
