import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check that Python version is at least 3.8."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required.")
        sys.exit(1)
    logger.info(f"Python version check passed: {sys.version}")

def install_requirements():
    """Install requirements from requirements.txt."""
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {str(e)}")
        sys.exit(1)

def download_spacy_model():
    """Download spaCy language model."""
    try:
        logger.info("Downloading spaCy language model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        logger.info("Successfully downloaded spaCy language model")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download spaCy model: {str(e)}")
        sys.exit(1)

def setup_directories():
    """Set up necessary directories for the project."""
    dirs = ["input", "output", "logs"]
    logger.info("Setting up directories...")
    
    for directory in dirs:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    # Create an example .env file if it doesn't exist
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write("""# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# PlantUML Configuration
PLANTUML_SERVER_URL=http://www.plantuml.com/plantuml/img/
""")
        logger.info(f"Created example .env file: {env_path}")

def main():
    """Main setup function."""
    logger.info("Starting UMLify setup...")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Download spaCy model
    download_spacy_model()
    
    # Setup directories
    setup_directories()
    
    logger.info("""
UMLify setup completed successfully!

Next steps:
1. Create a .env file with your API keys and Neo4j credentials
2. Place your SRS PDF files in the input/ directory
3. Run the main script: python src/main.py
""")

if __name__ == "__main__":
    main() 