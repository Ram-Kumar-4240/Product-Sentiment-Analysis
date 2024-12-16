from flask import Flask, request, jsonify, render_template
import joblib

# Load the saved model and tokenizer
model = joblib.load('Model and tokenizer//logistic_regression_model.pkl')
tokenizer = joblib.load('Model and tokenizer//tokenizer.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define the home route with a form for user input
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predicting sentiment
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form submission or API request
    if request.form:
        review_text = request.form.get('review')
    else:
        data = request.get_json()
        review_text = data.get('review')
    
    if not review_text:
        return jsonify({'error': 'No review text provided'}), 400

    # Preprocess and predict
    try:
        # Tokenize and transform input text
        tokenized_review = tokenizer.transform([review_text])
        
        # Predict sentiment
        prediction = model.predict(tokenized_review)
        sentiment = ['Negative', 'Neutral', 'Positive'][prediction[0]]

        # If the request is from the form, return HTML response
        if request.form:
            return render_template('index.html', sentiment=sentiment, review=review_text)

        # For API requests, return JSON response
        return jsonify({'review': review_text, 'sentiment': sentiment})
    except Exception as e:
        if request.form:
            return render_template('index.html', error=str(e))
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
