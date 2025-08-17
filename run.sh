#!/bin/bash

# Bizz | Docs Startup Script
echo "🔹 Starting Bizz | Docs..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set."
    echo "   Please set it before running the application:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "   Or create a .env file with your API key."
    echo ""
fi

# Start the application
echo "🚀 Starting Streamlit application..."
echo "   Open your browser and go to: http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
