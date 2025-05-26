import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants and configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
TEXT_FILE_PATH = os.getenv("TEXT_FILE_PATH", "../input/dineout.txt")

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set in your environment variables")
    
print("OpenRouter configuration: FREE tier")
print("Rate limits: 20 requests per minute, 50 requests per day")
print("IMPORTANT: Efficiently batch your requests to avoid hitting daily limits") 