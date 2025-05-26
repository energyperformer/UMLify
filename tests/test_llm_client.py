import os
import sys
import json
import time
from dotenv import load_dotenv
from llm_client import LLMClient

def test_api_connection():
    """Test basic API connectivity with a simple prompt."""
    print("=== Testing API Connection ===")
    client = LLMClient()
    
    # Override validate_response to always return True
    client.validate_response = lambda x: True
    
    # Simple prompt to test connectivity
    prompt = "Say 'Hello, world!' and nothing else."
    
    try:
        response = client.generate_text(prompt, max_tokens=20, temperature=0.1)
        print(f"Response received. Length: {len(response) if response else 0}")
        print(f"Response content: '{response}'")
        print("Connection test: SUCCESS\n")
    except Exception as e:
        print(f"Connection test: FAILED")
        print(f"Error: {str(e)}\n")
        if hasattr(e, 'response'):
            print(f"Error response: {e.response}\n")

def test_validation_function():
    """Test the validation function with different inputs."""
    print("=== Testing Validation Function ===")
    client = LLMClient()
    
    test_cases = [
        ("", "Empty string"),
        ("Short", "Short string"),
        ("{}", "Empty JSON"),
        ('{"key": "value"}', "Simple JSON"),
        ("error: something went wrong", "Error pattern"),
        ("This is a normal response with enough text to pass validation", "Normal text")
    ]
    
    for test_input, description in test_cases:
        result = client.validate_response(test_input)
        print(f"Input: '{test_input[:30]}...' ({description})")
        print(f"Validation result: {'PASS' if result else 'FAIL'}\n")

def test_response_handling():
    """Test how the client handles different response types."""
    print("=== Testing Response Handling ===")
    client = LLMClient()
    
    # Override validate_response to see raw responses
    original_validation = client.validate_response
    client.validate_response = lambda x: True
    
    prompts = [
        ("What is 2+2? Answer with just the number.", "Simple math"),
        ("Return a JSON with keys 'name' and 'age'", "JSON generation"),
        ("Tell me about the Python programming language in one sentence", "Longer text")
    ]
    
    for prompt, description in prompts:
        print(f"Testing: {description}")
        print(f"Prompt: '{prompt}'")
        
        try:
            response = client.generate_text(prompt, max_tokens=100, temperature=0.1)
            print(f"Response received. Length: {len(response) if response else 0}")
            print(f"Response content: '{response}'")
            print(f"Original validation would return: {original_validation(response)}")
            print("Test: SUCCESS\n")
        except Exception as e:
            print(f"Test: FAILED")
            print(f"Error: {str(e)}\n")
        
        # Wait between requests to avoid rate limiting
        time.sleep(5)

def main():
    print("===== LLM Client Diagnostic Tests =====")
    print(f"Running tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Make sure we have an API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)
    else:
        print(f"API Key found: {api_key[:3]}...{api_key[-3:]}\n")
    
    # Run tests
    test_api_connection()
    time.sleep(5)  # Add delay between tests
    test_validation_function()
    time.sleep(5)  # Add delay between tests
    test_response_handling()
    
    print("\n===== Diagnostics Complete =====")

if __name__ == "__main__":
    main() 