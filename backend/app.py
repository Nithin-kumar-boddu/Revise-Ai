
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
from .models.gemini import GeminiAPI
from .utils.cache import Cache

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

# Initialize Gemini API and cache
# Fix: Split the environment variable in case of inline comments, so that only the numeric part is used.
cache = Cache(expiration=int(os.getenv("CACHE_EXPIRATION", "3600").split()[0]))

@app.route('/explain', methods=['POST'])
def explain():
    """Endpoint for generating explanations of topics"""
    data = request.json
    topic = data.get('topic')
    
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    
    # Check cache first
    cache_key = f"explain_{topic}"
    cached_response = cache.get(cache_key)
    
    if cached_response:
        return jsonify({"explanation": cached_response})
    
    try:
        explanation = GeminiAPI(api_key=os.getenv("GEMINI_API_KEY")).generate_explanation(topic)
        # Cache the response
        cache.set(cache_key, explanation)
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint for summarizing text"""
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Check cache first using a hash since text can be long
    cache_key = f"summarize_{hash(text)}"
    cached_response = cache.get(cache_key)
    
    if cached_response:
        return jsonify({"summary": cached_response})
    
    try:
        summary = GeminiAPI(api_key=os.getenv("GEMINI_API_KEY")).generate_summary(text)
        # Cache the response
        cache.set(cache_key, summary)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
