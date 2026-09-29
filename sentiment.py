import spacy
from transformers import pipeline
nlp = spacy.load("en_core_web_sm")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
def preprocess_text(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_space:
            tokens.append(token.text)
    return " ".join(tokens)
def analyze_sentiment(text):
    cleaned_text = preprocess_text(text)
    result = sentiment_model(cleaned_text)[0]
    label = result["label"]
    confidence = result["score"]
    if label in ["1 star", "2 stars"]:
        sentiment = "Negative"
    elif label == "3 stars":
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    return sentiment, confidence