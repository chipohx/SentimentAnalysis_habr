from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from ml.model import load_model

model = None

class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    try:
        yield
    finally:
        pass

app = FastAPI(lifespan=lifespan)

@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}


@app.get("/predict")
def predict_sentiment(text: str):
    sentiment = model(text)

    response = SentimentResponse(
        text = text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )

    return response