import nltk
nltk.download('punkt')

import whisper
from nltk.tokenize import sent_tokenize
from transformers import pipeline

def summarize_podcast(audio_path):
    # Transcribe audio using Whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"]

    # Tokenize into sentences and group into chunks
    sentences = sent_tokenize(transcript)
    chunks, current = [], ""
    for sent in sentences:
        if len(current.split()) + len(sent.split()) < 200:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    # Summarize each chunk
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaries = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]

    return "\n\n".join(summaries)
