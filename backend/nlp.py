import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pypdf import PdfReader
import io

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def extract_companies_and_people(text):
    doc = nlp(text)
    companies = list(set(e.text for e in doc.ents if e.label_ == "ORG"))
    people = list(set(e.text for e in doc.ents if e.label_ == "PERSON"))
    return {"companies": companies, "people": people}

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pos = probs[0][0].item()
    neg = probs[0][1].item()
    neu = probs[0][2].item()
    score = pos - neg
    return {
        "score": round(score, 4),
        "positive": round(pos, 4),
        "neutral": round(neu, 4),
        "negative": round(neg, 4),
        "label": "Positive" if score > 0.2 else ("Negative" if score < -0.2 else "Neutral")
    }

def summarize_text(text, num_sentences=3):
    text = text.replace('\n', ' ').replace('\r', ' ')
    sentences = text.split('.')
    keywords = ['billion', 'million', 'revenue', 'profit', 'increased',
                'decreased', 'launch', 'future', 'outlook', 'quarter',
                'year', 'expect']
    important = []
    for sent in sentences:
        sent = sent.strip()
        if 40 < len(sent) < 300:
            if any(k in sent.lower() for k in keywords):
                important.append(sent)
            if len(important) >= num_sentences:
                break
    return '. '.join(important) + '.' if important else "No summary available."

def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return text
