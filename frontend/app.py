import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
BACKEND_URL = "http://localhost:5000"

# Set page config
st.set_page_config(
    page_title="Revise AI - Your Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS styling from styles.css file
css_file = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_file) as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Page header
st.markdown('<h1 class="main-header">Revise AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-powered study assistant</p>', unsafe_allow_html=True)

# Initialize session state for message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(f'<div class="message-container user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-container ai-message"><strong>Revise AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)

# Sidebar for mode selection
st.sidebar.title("Study Mode")
mode = st.sidebar.selectbox(
    "Choose what you need help with:",
    ["Explain a Topic", "Summarize Text"]
)

# Get user input based on the selected mode
if mode == "Explain a Topic":
    user_input = st.text_area("Enter a topic you want to understand:", height=100)
    endpoint = "/explain"
    payload_key = "topic"
else:  # Summarize Text
    user_input = st.text_area("Paste the text you want to summarize:", height=300)
    endpoint = "/summarize"
    payload_key = "text"

# Process user input
if st.button("Submit"):
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show loading spinner
        with st.spinner("Thinking..."):
            try:
                # Make request to backend
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json={payload_key: user_input}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract AI response
                    if endpoint == "/explain":
                        ai_response = data.get("explanation", "Sorry, I couldn't generate an explanation.")
                    else:
                        ai_response = data.get("summary", "Sorry, I couldn't generate a summary.")
                    
                    # Append the response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response
                    })

                    # Force re-rendering - FIXED LINE
                    st.experimental_rerun()
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About Revise AI")
st.sidebar.markdown("""
Revise AI helps you prepare for exams by:
- Explaining complex topics in simple terms
- Summarizing lengthy study material
- Adapting to your learning needs
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Google Gemini API")

# Additional functionality: Clear Chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()  # FIXED LINE - Force app refresh to clear chat

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:14px; color:gray;'>
    <strong>Disclaimer:</strong><br>
    - This model is entirely based on academic knowledge. Any real-world information, coding, etc., may not always be appropriate.<br>
    - AI models can make mistakes; be cautious before considering the information provided.
</div>
""", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-weight:bold;'>Revise AI © 2025</p>", unsafe_allow_html=True)