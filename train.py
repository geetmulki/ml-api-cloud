import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "I love this movie, it is fantastic!",
    "Absolutely terrible, worst experience ever.",
    "Great acting and wonderful plot.",
    "I hated every minute of it.",
    "A beautiful and inspiring story.",
    "Boring, dull, and a waste of time."
]

labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

joblib.dump(model, 'sentiment_model.joblib')
print("Model saved")