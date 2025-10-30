"""
Simple test using the native Google Generative AI library
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"API Key found: {api_key is not None}")
print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")

# Test with requests to see if the API key works
import requests

url = "https://generativelanguage.googleapis.com/v1beta/models"
headers = {
    "x-goog-api-key": api_key
}

try:
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Available models: {len(data.get('models', []))}")
        for model in data.get('models', [])[:5]:  # Show first 5 models
            print(f"  - {model.get('name', 'Unknown')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")