from flask import Flask, request, jsonify
import joblib
import os

# Elastic Beanstalk looks for variable named "application"
application = Flask(__name__)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.joblib")
model = joblib.load(model_path)


@application.route("/")
def home():
    return "Sentiment API is running ✅"


@application.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    prediction = model.predict([text])[0]

    return jsonify({
        "input_text": text,
        "sentiment_prediction": prediction,
        "model_version": "1.0"
    })


# Local testing only (Elastic Beanstalk ignores this block)
if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)