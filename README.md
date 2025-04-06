# Revise AI - AI-Powered Study Assistant

Revise AI is an intelligent study assistant designed to help students prepare for exams by providing detailed explanations of topics and summarizing lengthy study materials.

## Features

- **Explanation Mode**: Get comprehensive explanations of academic topics
- **Summarization Mode**: Convert lengthy text passages into concise summaries
- **User-friendly Interface**: Intuitive chat-like interface for easy interaction

## Project Architecture

- **Frontend**: Built with Streamlit
- **Backend**: Flask API with two main endpoints:
  - `/explain` - Generates explanations for topics
  - `/summarize` - Summarizes text content
- **AI Engine**: Powered by Google Gemini API

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (get one from [Google AI Studio](https://ai.google.dev/))

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/Nithin-kumar-boddu/Revise-Ai.git
   cd revise-ai
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Configure your environment:
   - Copy the `.env.example` file to `.env`
   - Add your Google Gemini API key to the `.env` file

### Running the Application

1. Start the Flask backend:

   ```
   python -m backend.app
   ```

2. Start the Streamlit frontend:

   ```
   streamlit run frontend/app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## Usage Guide

1. **Select Mode**:

   - Choose "Explain a Topic" to get explanations
   - Choose "Summarize Text" to summarize content

2. **Input Content**:

   - For explanations, enter a topic you want to learn about
   - For summaries, paste the text you want to summarize

3. **Submit**:
   - Click the "Submit" button to process your request
   - View the AI-generated response in the chat interface

## Project Structure

```
revise_ai/
├── .env                  # Environment variables (API keys)
├── backend/
│   ├── __init__.py
│   ├── app.py            # Flask application
│   ├── models/
│   │   └── gemini.py     # Gemini API integration
│   └── utils/
│       └── cache.py      # Caching mechanism
├── frontend/
│   └── app.py            # Streamlit application
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
Disclaimer:
By using **Revise AI**, you agree that any commercial use, distribution, or misuse of the model without prior written consent from the creators will be subject to legal action. Unauthorized use or exploitation of this model for business purposes or any activities that violate its intended educational purpose will be prosecuted accordingly.
