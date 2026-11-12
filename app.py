from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

# Initialize the app
app = Flask(__name__)
CORS(app)  # Allows our Android app to talk to this server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading AI models...")
try:
    with open(os.path.join(BASE_DIR, 'sms_model.pkl'), 'rb') as f:
        sms_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'sms_vectorizer.pkl'), 'rb') as f:
        sms_vectorizer = pickle.load(f)
    print("✅ SMS Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading SMS models: {e}")

# --- Home Route ---
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
    
    # Extract features using the saved vectorizer
    text_features = sms_vectorizer.transform([sms_text])
    
    # Predict using the saved model
    prediction = sms_model.predict(text_features)[0]
    probabilities = sms_model.predict_proba(text_features)[0]
    risk_score = int(probabilities[1] * 100) # Probability of being fraudulent
    
    # Determine classification
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
        
    # Find suspicious words
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
    
    try:
        with open(os.path.join(BASE_DIR, 'url_model.pkl'), 'rb') as f:
            url_model = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'url_vectorizer.pkl'), 'rb') as f:
            url_vectorizer = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"URL model not found: {e}"}), 500
        
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
    # Cloud platforms assign a port dynamically
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)