from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp import extract_companies_and_people, analyze_sentiment, summarize_text, extract_text_from_pdf
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"status": "backend running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_pdf(content)
    ner = extract_companies_and_people(text[:3000])
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 40][:20]
    scores = []
    for sent in sentences:
        try:
            scores.append(analyze_sentiment(sent)["score"])
        except:
            scores.append(0)
    summary = summarize_text(text)
    return {
        "characters": len(text),
        "ner": ner,
        "sentiment": {
            "average": round(float(np.mean(scores)), 4) if scores else 0,
            "scores": scores
        },
        "summary": summary
    }

class QuestionRequest(BaseModel):
    question: str
    context: str

@app.post("/ask")
def ask(req: QuestionRequest):
    q = req.question.lower()
    ctx = req.context
    if any(w in q for w in ["revenue", "profit", "billion", "million"]):
        answer = f"Based on the transcript: {ctx[:300]}"
    elif "sentiment" in q:
        answer = "Check the Sentiment tab for detailed scores."
    else:
        answer = f"From the transcript summary: {ctx[:300]}"
    return {"answer": answer}
