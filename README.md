# Political-bias-analyzer
Text classification for political bias detection

# AI-Based Text Political Bias Detection
Technologies: TF-IDF, Logistic Regression, Machine Learning, NLP, Python

Developed a machine learning-based Python text classification system to identify political bias in textual data. Work included:

Goal: Created an automated system that detects and classifies political bias in text.

Preprocessing Data: Applied Natural Language Processing (NLP) methods like cleaning of text, tokenization, and lemmatization for normalization of input data.

Feature Engineering: Applied TF-IDF vectorization to transform text data into numerical features to train the model.

Model Development: Developed a Logistic Regression classifier to examine and classify political bias in text.

Evaluation Metrics: Evaluated the performance of the model using accuracy scores, classification reports, confusion matrices, and cross-validation strategies.

Dataset: Used a labeled dataset (Political_Bias.csv) with text samples and corresponding bias labels.

Impact: The model can be used to predict political bias in text, helping researchers, journalists, and analysts identify trends in political bias in discourse.

This project reflects proficiency in NLP, text classification, and bias detection, and demonstrates good analytical and machine learning skills.



**Step by step explanation of the code**

**1. Importing Libraries:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

* `pandas (pd)`: Used for data manipulation and analysis, particularly for working with DataFrames (tables).
* `sklearn.model_selection.train_test_split`: Splits the dataset into training and testing sets.
* `sklearn.feature_extraction.text.TfidfVectorizer`: Converts text data into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.
* `sklearn.linear_model.LogisticRegression`: The machine learning model used for classification.
* `sklearn.metrics.classification_report`, `sklearn.metrics.confusion_matrix`: Used to evaluate the model's performance.
* `matplotlib.pyplot (plt)`: Used for creating plots and visualizations.
* `seaborn (sns)`: Built on top of matplotlib, provides a higher-level interface for creating statistical graphics.

**2. Loading the Dataset:**

```python
try:
    df = pd.read_csv("political_bias.csv")
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure the path is correct.")
    exit()
```

* This code attempts to read a CSV file named "political\_bias.csv" into a pandas DataFrame called `df`.
* The `try...except` block handles the case where the file is not found, printing an error message and exiting the program.

**3. Data Preprocessing:**

```python
print(df.columns)
print(df["Text"].isnull().sum())
df["Text"] = df['Text'].fillna('')
if 'Text' not in df.columns or 'Bias' not in df.columns:
    print("Error: Required columns 'Text' or 'Bias' not found in the dataset.")
    exit()
df = df.dropna(subset=['Text', 'Bias'])
X = df['Text']
y = df['Bias']
```

* `print(df.columns)`: Prints the column names of the DataFrame to help understand the dataset's structure.
* `print(df["Text"].isnull().sum())`: Prints the number of missing values in the "Text" column.
* `df["Text"] = df['Text'].fillna('')`: Replaces any missing values (NaN) in the "Text" column with empty strings.
* The code then checks if the "Text" and "Bias" columns exist, exiting if they don't.
* `df = df.dropna(subset=['Text', 'Bias'])`: Removes any rows where either the "Text" or "Bias" column contains missing values.
* `X = df['Text']`: Assigns the "Text" column (the input features) to the variable `X`.
* `y = df['Bias']`: Assigns the "Bias" column (the target labels) to the variable `y`.

**4. Splitting the Data:**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* This splits the dataset into training and testing sets.
* `X_train` and `y_train` are used to train the model.
* `X_test` and `y_test` are used to evaluate the model's performance.
* `test_size=0.2` means 20% of the data is used for testing.
* `random_state=42` ensures that the split is reproducible.

**5. Text Vectorization (TF-IDF):**

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

* `TfidfVectorizer`: Converts the text data into numerical vectors that the model can understand.
* `max_features=5000`: Limits the number of features (words) to the top 5000 most frequent.
* `fit_transform(X_train)`: Learns the vocabulary and transforms the training data.
* `transform(X_test)`: Transforms the testing data using the vocabulary learned from the training data.

**6. Model Training (Logistic Regression):**

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
```

* `LogisticRegression()`: Creates a logistic regression model.
* `max_iter=1000`: increases the maximum number of iterations for the solver to converge.
* `model.fit(X_train_vec, y_train)`: Trains the model using the vectorized training data and the training labels.

**7. Model Evaluation:**

```python
y_pred = model.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", model.score(X_test_vec, y_test))
```

* `model.predict(X_test_vec)`: Uses the trained model to predict the labels for the testing data.
* `classification_report(y_test, y_pred)`: Prints a report containing precision, recall, F1-score, and support for each class.
* `model.score(X_test_vec, y_test)`: Prints the accuracy of the model.

**8. Confusion Matrix Visualization:**

```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```

* This code generates and displays a confusion matrix, which helps visualize the model's performance in classifying each class.

**9. Example Prediction Function:**

```python
def predict_bias(speech):
    speech_vec = vectorizer.transform([speech])
    prediction = model.predict(speech_vec)
    return prediction[0]

#example predictions.
```

* This function takes a speech as input, vectorizes it using the trained `vectorizer`, and predicts the bias using the trained `model`.
* The example predictions at the end of the code, show how the function can be used.
