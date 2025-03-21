import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #changed from CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score #added cross val
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #added metrics
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources if not already available
try:
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Political_Bias.csv", encoding="utf-8")

# ... (rest of the data loading and preprocessing code) ...

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2)) #changed to Tfidf and added ngram
X = vectorizer.fit_transform(df['Processed_Text'])

y = df['Bias']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000) #increased max_iter
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Cross Validation.
scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross Validation Scores: {scores}")
print(f"Mean Cross Validation Score: {scores.mean()}")