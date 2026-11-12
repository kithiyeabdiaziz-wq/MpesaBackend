import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading dataset...")
# 1. Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the first two columns (v1 = label, v2 = message text)
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels: 'ham' becomes 0, 'spam' becomes 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Splitting data...")
# 2. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print("Extracting features using TF-IDF...")
# 3. Text Preprocessing & Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training the AI Model...")
# 4. Train the Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Evaluating model...")
# 5. Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save the model and vectorizer to files
joblib.dump(model, 'sms_model.pkl')
joblib.dump(vectorizer, 'sms_vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")