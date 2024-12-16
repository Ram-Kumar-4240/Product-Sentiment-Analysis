import flask
from flask import Flask, request, render_template
import joblib
import numpy as np
import traceback

# Load the saved model and tokenizer
try:
    model = joblib.load('Model and tokenizer/logistic_regression_model.pkl')
    tokenizer = joblib.load('Model and tokenizer/tokenizer.pkl')
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    model = None
    tokenizer = None

# Initialize the Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for analyzing the sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the review text from the form
    review_text = request.form.get('review', '').strip()

    if not review_text:
        return render_template('index.html', error="No review text provided. Please enter a review.")

    try:
        if model is None or tokenizer is None:
            return render_template('index.html', error="Model or tokenizer not loaded correctly.")

        # Tokenize and vectorize the input text
        tokenized_review = tokenizer.transform([review_text])

        # Check the type of prediction method
        if hasattr(model, 'predict_proba'):
            # If predict_proba exists, use it
            prediction_proba = model.predict_proba(tokenized_review)
            sentiment_index = int(np.argmax(prediction_proba))
        else:
            # Fallback to predict method
            prediction = model.predict(tokenized_review)
            sentiment_index = int(prediction[0])  # Ensure it's an integer

        # Map the index to the sentiment label
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        
        # Ensure sentiment_index is within the range of labels
        sentiment_index = max(0, min(sentiment_index, len(sentiment_labels) - 1))
        sentiment = sentiment_labels[sentiment_index]

        # Print debug information
        print(f"Review: {review_text}")
        print(f"Tokenized Review Shape: {tokenized_review.shape}")
        print(f"Sentiment Index: {sentiment_index}")
        print(f"Predicted Sentiment: {sentiment}")

        # Render results back to the HTML template
        return render_template('index.html', 
                               sentiment=sentiment, 
                               review=review_text)

    except Exception as e:
        # Log the full traceback for debugging
        print("Full Error Traceback:")
        traceback.print_exc()
        return render_template('index.html', error=f"Error occurred: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)