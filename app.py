from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

lr_model = joblib.load('Model and tokenizer\\logistic_regression_model.pkl')  # Load the saved Logistic Regression model
tokenizer = joblib.load('Model and tokenizer\\tokenizer.pkl')  # Load the saved tokenizer

app = Flask(__name__)

def text_clean_1(text):

    text = text.lower()  
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    cleaned_review = text_clean_1(data)
    
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')  # Use max_len=200 from your model

    # Predict sentiment using the Logistic Regression model
    sentiment_class = lr_model.predict(padded_sequence)[0]
    
    # Sentiment mapping for classes (adjust if necessary based on your model output)
    sentiment_mapping = {-2: 'very negative', -1: 'negative', 0: 'neutral', 1: 'positive', 2: 'very positive'}
    sentiment = sentiment_mapping[sentiment_class]
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
