import os
from groq import Groq
from dotenv import load_dotenv

# 1. Load the secret .env file to get your key
load_dotenv()

# Check if key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env file.")
    print("Please make sure you added GROQ_API_KEY=gsk_... to your .env file.")
    exit()

# 2. Configure Groq
client = Groq(api_key=api_key)

print(f"✅ Key found: {api_key[:10]}...") # Print first few chars to verify
print("\nChecking available Groq models...\n")

try:
    # 3. List models
    models = client.models.list()
    
    # Print them nicely
    for model in models.data:
        print(f"- {model.id}")
        
    print("\n✨ Recommendation: Use 'llama3-8b-8192' for speed or 'llama3-70b-8192' for intelligence.")

except Exception as e:
    print(f"❌ Connection Error: {e}")