import os
import sys
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

def test_api_connection():
    """Test basic API connectivity with minimal prompt."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        return False
        
    logger.info("API key loaded successfully")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Very simple prompt to test connectivity
    test_prompt = "Say 'Hello World' and nothing else."
    
    try:
        logger.info("Testing API connection with simple prompt...")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "UMLify"
            },
            model="meta-llama/llama-4-scout:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": test_prompt
                }
            ],
            max_tokens=20,
            temperature=0.1
        )
        
        if not completion or not hasattr(completion, 'choices'):
            logger.error("Empty or invalid completion object")
            return False
            
        content = completion.choices[0].message.content
        logger.info(f"API Response: '{content}'")
        
        # Wait for rate limit to reset
        logger.info("Waiting 10 seconds before next test...")
        time.sleep(10)
        
        return True
        
    except Exception as e:
        logger.error(f"API connection test failed: {str(e)}")
        return False

def test_longer_prompt():
    """Test with a slightly longer prompt to see if size is the issue."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Medium-sized prompt 
    test_prompt = "Summarize the following paragraph in 2-3 sentences: " + (
        "The use case diagram is a graphical overview of the functionality provided by a system in terms " +
        "of actors, their goals represented as use cases, and any dependencies between those use cases. " +
        "In the Unified Modeling Language, a use case diagram can be used to summarize the roles of users " +
        "who interact with the system and the specific functionality or services the system provides to " +
        "those users. Use case diagrams are typically created early in the development process to understand " +
        "and document the functional requirements of the system."
    )
    
    try:
        logger.info("Testing API with medium-length prompt...")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "UMLify"
            },
            model="meta-llama/llama-4-scout:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": test_prompt
                }
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        content = completion.choices[0].message.content
        logger.info(f"API Response: '{content}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Medium prompt test failed: {str(e)}")
        return False

def main():
    logger.info("=== API DIAGNOSTIC TESTS ===")
    
    # Test basic connectivity
    if test_api_connection():
        logger.info("✓ Basic API connectivity test passed")
    else:
        logger.error("✗ Basic API connectivity test failed")
        return
    
    # Test with longer prompt
    if test_longer_prompt():
        logger.info("✓ Medium-length prompt test passed")
    else:
        logger.error("✗ Medium-length prompt test failed")
    
    logger.info("=== DIAGNOSTIC TESTS COMPLETE ===")

if __name__ == "__main__":
    main() 