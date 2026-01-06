import os
from openai import OpenAI

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"

TEMPERATURE = 0.5

# Initialize OpenAI client (same setup as updated_workflow.py)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Make a simple call to nemotron to check default temperature
print("Checking Nemotron default temperature...")
print(f"Model: openrouter/nvidia/nemotron-3-nano-30b-a3b")
print(f"Base URL: {BASE_URL}")
print("-" * 60)

try:
    # Make a simple test call (same as in updated_workflow.py)
    response = client.responses.create(
        # model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
        model="gpt-5",
        input="What is 2+2?",
        temperature=TEMPERATURE
    )
    
    # Check if response object has temperature attribute
    print("\nResponse object attributes:")
    print(f"Available attributes: {dir(response)}")
    
    # Try to access temperature if available
    if hasattr(response, 'temperature'):
        print(f"\n✅ Default Temperature: {response.temperature}")
    elif hasattr(response, 'model_info'):
        print(f"\nModel Info: {response.model_info}")
    else:
        print("\n⚠️  Temperature not directly available in response object")
        print("   Checking response metadata...")
        
        # Check if there's metadata or other attributes
        if hasattr(response, 'metadata'):
            print(f"   Metadata: {response.metadata}")
        if hasattr(response, 'usage'):
            print(f"   Usage: {response.usage}")
        if hasattr(response, 'model'):
            print(f"   Model: {response.model}")
    
    print(f"\n✅ Response received successfully")
    print(f"   Output text length: {len(response.output_text)} characters")
    print(f"   Output preview: {response.output_text[:100]}...")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"   Error type: {type(e).__name__}")

