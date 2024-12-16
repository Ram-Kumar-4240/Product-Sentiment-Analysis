# sentiment analysis on amazon product reviews

It seems like you're in the process of training several models for sentiment analysis on a dataset of Amazon reviews. Here's a breakdown of what you've done so far:

## Data Collection and Preprocessing:
- Loaded the Amazon reviews dataset and kept only the Text and Sentiment columns.
- Cleaned the text by removing punctuation, numbers, and special characters.
- Since the dataset contains several columns, you're keeping only the relevant ones: Text and Sentiment. This makes the data more manageable.

## Text Cleaning
### Text Cleaning Function:
- The text cleaning process involves converting all text to lowercase, removing punctuation, numbers, and special characters, as well as cleaning up any unnecessary content like text within square brackets (often metadata or links).

```python
  def text_clean_1(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove content inside square brackets
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)  # Remove numbers
    return text

```
You apply this function to the Text column, creating a new column Cleaned_Text that contains the cleaned text.

## Sentiment Distribution and Merging Classes
#### 1. Visualizing Sentiment Distribution:
- You visualize the distribution of sentiment labels using a countplot from seaborn. This gives you a clear view of how many reviews belong to each sentiment category.
#### 2. Merging Sentiment Classes:
- The sentiment labels were initially more granular, but you decide to merge similar categories for simplification:
  - Very Negative and Negative are combined into Negative.
  - Positive and Very Positive are combined into Positive.
This results in only three sentiment categories: Negative, Neutral, and Positive.


## Balancing the Dataset
#### 1. Handling Class Imbalance:
- detect and handle class imbalance by randomly sampling from each sentiment class. You use a maximum of 8000 samples per class to balance the dataset. After sampling, you concatenate the classes and shuffle the dataset to ensure randomness.
#### 2. Visualizing Balanced Distribution:
- After balancing, you again visualize the sentiment distribution to confirm that the classes are now equally represented.


## Feature Extraction:
#### 1. Vectorizing Text Using TF-IDF:
- To convert the cleaned text data into numerical form, you use TF-IDF (Term Frequency-Inverse Document Frequency) with a maximum of 5000 features. This step transforms the text data into a matrix of numerical values suitable for machine learning models.

```python
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(balanced_data['Cleaned_Text']).toarray()
```
Here, X is the feature matrix (i.e., the TF-IDF representation of the cleaned text), and y is the target variable (sentiment).


## Model Training and Evaluation

#### 1. Train-Test Split:
- You split the dataset into training and testing sets using train_test_split(). The training set is used to train the models, while the test set is used to evaluate their performance.
#### 2. Training Multiple Models:
- Logistic Regression: A simple linear model for binary or multi-class classification.
- Naive Bayes: A probabilistic model based on Bayes' theorem, typically effective for text classification tasks.
- Random Forest: An ensemble method that combines multiple decision trees to improve classification accuracy.
- XGBoost: An optimized gradient boosting framework that performs well for structured/tabular data like text classification.
#### 3. Model Evaluation:
- Accuracy: The percentage of correct predictions.
- Confusion Matrix: A matrix that shows the true positives, true negatives, false positives, and false negatives.
- Classification Report: Includes precision, recall, and F1-score for each class.

```python
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n{model_name} Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

```
#### 4. XGBoost
- You use XGBClassifier from the xgboost library for the final model, where the labels are encoded using LabelEncoder.
The predictions are made and then decoded back into their original categorical labels.

## Advanced Model (Deep Learning)

You also seem to have started preparing for a deep learning model with TensorFlow/Keras. The next steps involve:

- Tokenizing and padding the text data for neural network compatibility.
- Using a neural network model (like LSTM or CNN) for improved text classification performance.

This is a good progression as you move from traditional machine learning algorithms to deep learning models, which can capture more complex relationships in the text data.

## Conclusion
I gone through data preprocessing, cleaning, and balancing steps and trained several machine learning models for sentiment classification. Your next steps likely involve further tuning the models, possibly improving performance with deep learning techniques, and evaluating the final results.

## Contributors
- Name : Ramkumar
- Email : infogramrk@gmail.com

