"""
Test script to check available Gemini models with your API key
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No GEMINI_API_KEY found in environment variables")
    exit(1)

print(f"✅ Found API key: {api_key[:10]}...")

# Configure the API
genai.configure(api_key=api_key)

print("\n🔍 Available models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  ✅ {model.name}")
        else:
            print(f"  ❌ {model.name} (doesn't support generateContent)")
except Exception as e:
    print(f"❌ Error listing models: {e}")

# Test a simple generation
print("\n🧪 Testing model access...")
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, how are you?")
    print(f"✅ gemini-pro works: {response.text[:50]}...")
except Exception as e:
    print(f"❌ gemini-pro failed: {e}")

try:
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content("Hello, how are you?")
    print(f"✅ gemini-1.5-pro works: {response.text[:50]}...")
except Exception as e:
    print(f"❌ gemini-1.5-pro failed: {e}")

try:
    model = genai.GenerativeModel('gemini-1.0-pro')
    response = model.generate_content("Hello, how are you?")
    print(f"✅ gemini-1.0-pro works: {response.text[:50]}...")
except Exception as e:
    print(f"❌ gemini-1.0-pro failed: {e}")