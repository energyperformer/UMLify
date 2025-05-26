import os
import sys
import logging
from dotenv import load_dotenv

def test_environment():
    """Test environment variables and dependencies."""
    print("Testing environment setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        "OPENROUTER_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "PLANTUML_SERVER_URL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment variables loaded")
    return True

def test_dependencies():
    """Test required Python packages."""
    print("\nTesting dependencies...")
    
    required_packages = [
        "python-dotenv",
        "neo4j",
        "PyPDF2",
        "plantuml",
        "openai",
        "requests",
        "typing-extensions",
        "python-json-logger",
        "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not installed")
    
    if missing_packages:
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_neo4j_connection():
    """Test Neo4j connection."""
    print("\nTesting Neo4j connection...")
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print("✅ Neo4j connection successful")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {str(e)}")
        return False

def test_plantuml():
    """Test PlantUML setup."""
    print("\nTesting PlantUML...")
    try:
        import plantuml
        server_url = os.getenv("PLANTUML_SERVER_URL")
        plantuml.PlantUML(url=server_url)
        print("✅ PlantUML setup successful")
        return True
    except Exception as e:
        print(f"❌ PlantUML setup failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Starting setup tests...\n")
    
    tests = [
        ("Environment Variables", test_environment),
        ("Dependencies", test_dependencies),
        ("Neo4j Connection", test_neo4j_connection),
        ("PlantUML Setup", test_plantuml)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n=== Testing {name} ===")
        result = test_func()
        results.append((name, result))
    
    print("\n=== Test Results ===")
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! The environment is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 