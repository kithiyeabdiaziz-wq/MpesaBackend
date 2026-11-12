from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# Initialize the app
app = Flask(__name__)
CORS(app)  # Allows our Android app to talk to this server

# Load the trained AI Model and Vectorizer
print("Loading AI models...")
try:
    sms_model = joblib.load('sms_model.pkl')
    sms_vectorizer = joblib.load('sms_vectorizer.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure you ran train_sms_model.py first!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "M-Pesa Phishing Detection API is running!"})

# --- SMS Analysis Endpoint ---
@app.route('/api/analyze/sms', methods=['POST'])
def analyze_sms():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    sms_text = data['text']
    
    # 1. Extract features using the saved vectorizer
    text_features = sms_vectorizer.transform([sms_text])
    
    # 2. Predict using the saved model
    # 1 = Spam/Fraudulent, 0 = Ham/Safe
    prediction = sms_model.predict(text_features)[0]
    
    # 3. Get probability (confidence) score
    # predict_proba returns [[prob_safe, prob_fraud]]
    probabilities = sms_model.predict_proba(text_features)[0]
    risk_score = int(probabilities[1] * 100) # Probability of being fraudulent
    
    # 4. Determine classification based on risk score
    if risk_score >= 70:
        classification = "Fraudulent"
        recommendation = "Do not reply or click any links. Delete this message immediately."
        explanation = "The AI detected strong patterns typical of M-Pesa phishing scams."
    elif risk_score >= 30:
        classification = "Suspicious"
        recommendation = "Be cautious. Verify with Safaricom directly before taking action."
        explanation = "The AI detected some unusual patterns, but is not fully certain."
    else:
        classification = "Safe"
        recommendation = "This message appears normal. Stay vigilant."
        explanation = "No common phishing patterns were detected."
        
    # 5. Find suspicious words (Simple keyword matching for explanation)
    suspicious_keywords = ["mpesa", "update", "account", "suspended", "click", "link", "verify", "pin", "blocked", "reward"]
    found_words = [word for word in suspicious_keywords if word in sms_text.lower()]
    
    return jsonify({
        "classification": classification,
        "risk_score": risk_score,
        "suspicious_words": found_words,
        "explanation": explanation,
        "recommendation": recommendation
    })

# --- URL Analysis Endpoint ---
@app.route('/api/analyze/url', methods=['POST'])
def analyze_url():
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
        
    url_text = data['url']
    
    # Load URL model
    import os
    if not os.path.exists('url_model.pkl'):
        return jsonify({"error": "URL model not trained yet."}), 500
        
    url_model = joblib.load('url_model.pkl')
    url_vectorizer = joblib.load('url_vectorizer.pkl')
    
    # Extract features and predict
    url_features = url_vectorizer.transform([url_text])
    prediction = url_model.predict(url_features)[0]
    probabilities = url_model.predict_proba(url_features)[0]
    risk_score = int(probabilities[1] * 100)
    
    if risk_score >= 70:
        classification = "Fraudulent"
        recommendation = "Do not open this URL."
    elif risk_score >= 30:
        classification = "Suspicious"
        recommendation = "Proceed with caution."
    else:
        classification = "Safe"
        recommendation = "This URL appears safe."
        
    return jsonify({
        "classification": classification,
        "risk_score": risk_score,
        "explanation": "URL analyzed using character patterns and suspicious domain indicators.",
        "recommendation": recommendation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)