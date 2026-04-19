from flask import Flask, request, jsonify
import joblib
import os

application = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.joblib')

model = joblib.load(model_path)

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction,
        'model_version': '1.0'
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)