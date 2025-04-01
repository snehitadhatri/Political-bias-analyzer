import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with your Kaggle dataset path)
try:
    df = pd.read_csv("political_speech_bias.csv")  # Example filename
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure the path is correct.")
    exit()

# Preprocessing (adjust based on your dataset's columns)
if 'text' not in df.columns or 'bias' not in df.columns:
    print("Error: Required columns 'text' or 'bias' not found in the dataset.")
    exit()

df = df.dropna(subset=['text', 'bias'])  # Remove rows with missing values
X = df['text']
y = df['bias']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training (Logistic Regression)
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train_vec, y_train)

# Model Evaluation
y_pred = model.predict(X_test_vec)

# Output Results
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#Example prediction.
example_speech = ["The other party is completely wrong and is destroying our country. We must take action against them."]
example_speech_vec = vectorizer.transform(example_speech)
prediction = model.predict(example_speech_vec)
print(f"Example Speech: {example_speech[0]}")
print(f"Predicted Bias: {prediction[0]}")

example_speech2 = ["We must work together to find common ground and solve our problems."]
example_speech_vec2 = vectorizer.transform(example_speech2)
prediction2 = model.predict(example_speech_vec2)
print(f"Example Speech: {example_speech2[0]}")
print(f"Predicted Bias: {prediction2[0]}")
