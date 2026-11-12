import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Create a sample dataset of Safe and Fraudulent URLs
print("Generating URL dataset...")
safe_urls = [
    "https://www.safaricom.co.ke", "https://www.mpesa.com", "https://google.com", 
    "https://facebook.com", "https://amazon.com", "https://wikipedia.org",
    "https://github.com", "https://stackoverflow.com", "https://youtube.com", "https://twitter.com"
]
fraud_urls = [
    "http://safaricom-mpesa-update.com", "http://192.168.1.1/mpesa/login", 
    "http://verify-mpesa-account.xyz", "http://mpesa-login-secure.top", 
    "http://bit.ly/3xYz", "http://free-mpesa-money.ga", "http://safaricom.mpesa.verify.cf",
    "http://mpesa-pin-reset.buzz", "http://click-here-to-win-mpesa.tk", "http://10.0.0.5/login"
]

# Multiply to create a larger dataset for training
data = []
for _ in range(50):
    for url in safe_urls:
        data.append({"url": url, "label": 0}) # 0 = Safe
    for url in fraud_urls:
        data.append({"url": url, "label": 1}) # 1 = Fraudulent

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(df['url'], df['label'], test_size=0.2, random_state=42)

# 3. Extract features (using Character-level TF-IDF to catch suspicious keywords/patterns)
print("Training URL model...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_tfidf)
print(f"URL Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Save the model
joblib.dump(model, 'url_model.pkl')
joblib.dump(vectorizer, 'url_vectorizer.pkl')
print("✅ URL Model saved successfully!")