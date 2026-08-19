"""
SMS Spam Detection - Flask Web Application
Provides a web interface to test SMS messages for spam classification.
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuration
MODEL_DIR = './model'
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Load model and vectorizer at startup
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("[OK] Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("[ERROR] Model files not found!")
    print("  Please run: python train_model.py")
    model = None
    vectorizer = None


@app.route('/')
def home():
    """Render the homepage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict if a message is spam or ham.
    Expects JSON: {"message": "user message"}
    Returns JSON: {"prediction": "SPAM/HAM", "message": "full result message"}
    """
    
    # Validation
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 500
    
    # Get message from request
    data = request.get_json()
    message = data.get('message', '').strip()
    
    # Validate input
    if not message:
        return jsonify({
            'error': 'Please enter a message.'
        }), 400
    
    # Check message length
    if len(message) > 1000:
        return jsonify({
            'error': 'Message too long (max 1000 characters).'
        }), 400
    
    try:
        # Vectorize the message
        message_tfidf = vectorizer.transform([message])
        
        # Make prediction (0 = ham, 1 = spam)
        prediction = model.predict(message_tfidf)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(message_tfidf)[0]
        confidence = max(probabilities) * 100
        
        # Format result
        if prediction == 0:
            result = "HAM"
            message_text = "This message appears safe. ✓"
        else:
            result = "SPAM"
            message_text = "This message is likely spam. ⚠"
        
        return jsonify({
            'prediction': result,
            'message': message_text,
            'confidence': f"{confidence:.1f}%"
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    if model is None or vectorizer is None:
        print("\n" + "="*60)
        print("ERROR: Model not found!")
        print("="*60)
        print("\nPlease train the model first:")
        print("  python train_model.py")
        print("\n" + "="*60)
    else:
        print("\n" + "="*60)
        print("SMS SPAM DETECTION - FLASK SERVER")
        print("="*60)
        print("\n*** Server running at http://localhost:5000")
        print("   Press Ctrl+C to stop")
        print("\n" + "="*60)
        app.run(debug=True, host='127.0.0.1', port=5000)
