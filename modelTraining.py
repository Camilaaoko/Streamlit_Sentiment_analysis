import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords, words as nltk_words
import nltk
import os
import glob
import datetime
import pandas as pd


nltk.download('stopwords')
nltk.download('punkt')  
stop_words = set(stopwords.words('english'))
english_vocab = set(nltk_words.words())  # Load dictionary words

# Load dataset
df = pd.read_csv('C:\\Users\\dell\\Tweets.csv', 
                 encoding='ISO-8859-1')
print(df.columns)
print(df['airline_sentiment'].value_counts())
df[['airline_sentiment', 'text']].head()

# Preprocessing function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)#remove numbers
    text = re.sub(r'\b[^aeiou\s]{4,}\b', '', text)# Remove words without vowels
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    negations = {"don't", "not", "never", "no", "wouldn't", "can't", "shouldn't"}
    words = text.split()
    words = [word for word in words if word not in stop_words or word in negations]
    meaningful_words = {"i", "no", "up", "ok", "go"}  # Essential short words
   
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [word for word in words if len(word) > 2 or word in meaningful_words]  # Remove short words
    words = [word for word in words if word in english_vocab or word in meaningful_words]  # Remove gibberish
    for i, word in enumerate(words):
        if word.lower() in negations and i + 1 < len(words):
            words[i + 1] = "not_" + words[i + 1]  # Transform "wouldn't recommend" to "not_recommend"
    return " ".join(words)
   
def convert_numerical_ratings(text):
    ratings_map = {
        "10": " very positive ",  # Note: Added spaces
        "9": " very positive ",
        "8": " positive ",
        "7": " positive ",
        "6": " slightly positive ",
        "5": " neutral ",
        "4": " slightly negative ",
        "3": " negative ",
        "2": " very negative ",
        "1": " very negative ",
    }
    for num, sentiment in ratings_map.items():
        text = text.replace(num, sentiment)  # Replace regardless of surrounding chars
    return text



df['cleaned_text'] = df['text'].apply(lambda x: clean_text(convert_numerical_ratings(x)))

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(label_map)
# Train-test split
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3),stop_words=None )
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(class_weight='balanced', random_state=42, multi_class='multinomial')
model.fit(X_train_tfidf, y_train)
X_test_tfidf = vectorizer.transform(X_test)


# Save model and vectorizer

save_dir = './APP'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


# Save the model and vectorizer to the specified directory

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_path = os.path.join(save_dir, f'sentiment_model_{timestamp}.pkl')
vectorizer_path = os.path.join(save_dir, f'vectorizer_{timestamp}.pkl')
print(f"Saving model at: {model_path}")
print(f"Saving vectorizer at: {vectorizer_path}")


try:
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved at: {model_path}")
    print(f"Vectorizer saved at: {vectorizer_path}")
except Exception as e:
    print(f"Error saving files: {e}")

example_texts = [
    "I absolutely love this! Best day ever!",
    "This is terrible, worst experience of my life",
    "Just got a new job! So excited!",
    "Lost my phone today :(",
    "This phone is a 10/10"
]


example_processed = [convert_numerical_ratings(text) for text in example_texts]
example_processed = [clean_text(text) for text in example_texts]
example_tfidf = vectorizer.transform(example_processed)
example_predictions = model.predict(example_tfidf)
sentiment_map = {0: "Negative", 2: "Positive", 1: "Neutral"}

print("\
Example Predictions:")
for text, pred in zip(example_texts, example_predictions):
    print(f"Text: {text}")
    print(f"Predicted sentiment: {sentiment_map[pred]}\n") 