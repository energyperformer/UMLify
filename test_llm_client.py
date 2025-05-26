#!/usr/bin/env python3
"""
Test file for LLMClient to verify OpenRouter API integration.
Tests basic functionality, provider options, rate limiting, and error handling.
"""

import os
import sys
import time
import json
from dotenv import load_dotenv

# Add src directory to path to import LLMClient
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_client import LLMClient

def test_basic_functionality():
    """Test basic text generation functionality."""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)
    
    try:
        client = LLMClient()
        print("✓ LLMClient initialized successfully")
        
        # Simple test prompt
        prompt = "What is UML? Give a brief 2-sentence answer."
        print(f"Prompt: {prompt}")
        
        response = client.generate_text(prompt, max_tokens=100, temperature=0.2)
        print(f"Response: {response}")
        print(f"Response length: {len(response)} characters")
        
        if response and len(response) > 10:
            print("✓ Basic functionality test PASSED")
            return True
        else:
            print("✗ Basic functionality test FAILED - Response too short")
            return False
            
    except Exception as e:
        print(f"✗ Basic functionality test FAILED: {str(e)}")
        return False

def test_provider_order():
    """Test provider order functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Provider Order")
    print("=" * 60)
    
    try:
        client = LLMClient()
        
        # Test with provider order specified
        prompt = "Say 'Hello from Llama model' in exactly those words."
        print(f"Prompt: {prompt}")
        
        response = client.generate_text(
            prompt, 
            max_tokens=50, 
            temperature=0.1,
            provider_order=["meta"],
            allow_fallbacks=True
        )
        print(f"Response: {response}")
        
        if response:
            print("✓ Provider order test PASSED")
            return True
        else:
            print("✗ Provider order test FAILED - No response")
            return False
            
    except Exception as e:
        print(f"✗ Provider order test FAILED: {str(e)}")
        return False

def test_fallback_options():
    """Test fallback options functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Fallback Options")
    print("=" * 60)
    
    try:
        client = LLMClient()
        
        # Test with fallbacks enabled (default behavior)
        prompt = "Count from 1 to 3."
        print(f"Prompt: {prompt}")
        
        response = client.generate_text(
            prompt, 
            max_tokens=30, 
            temperature=0.1,
            allow_fallbacks=True
        )
        print(f"Response: {response}")
        
        if response:
            print("✓ Fallback options test PASSED")
            return True
        else:
            print("✗ Fallback options test FAILED - No response")
            return False
            
    except Exception as e:
        print(f"✗ Fallback options test FAILED: {str(e)}")
        return False

def test_parameter_validation():
    """Test input parameter validation."""
    print("\n" + "=" * 60)
    print("TEST 4: Parameter Validation")
    print("=" * 60)
    
    client = LLMClient()
    tests_passed = 0
    total_tests = 4
    
    # Test empty prompt
    try:
        client.generate_text("", max_tokens=100)
        print("✗ Empty prompt validation FAILED - Should have raised error")
    except ValueError:
        print("✓ Empty prompt validation PASSED")
        tests_passed += 1
    
    # Test negative max_tokens
    try:
        client.generate_text("Hello", max_tokens=-1)
        print("✗ Negative max_tokens validation FAILED - Should have raised error")
    except ValueError:
        print("✓ Negative max_tokens validation PASSED")
        tests_passed += 1
    
    # Test invalid temperature (too high)
    try:
        client.generate_text("Hello", temperature=1.5)
        print("✗ High temperature validation FAILED - Should have raised error")
    except ValueError:
        print("✓ High temperature validation PASSED")
        tests_passed += 1
    
    # Test invalid temperature (negative)
    try:
        client.generate_text("Hello", temperature=-0.1)
        print("✗ Negative temperature validation FAILED - Should have raised error")
    except ValueError:
        print("✓ Negative temperature validation PASSED")
        tests_passed += 1
    
    if tests_passed == total_tests:
        print(f"✓ Parameter validation test PASSED ({tests_passed}/{total_tests})")
        return True
    else:
        print(f"✗ Parameter validation test FAILED ({tests_passed}/{total_tests})")
        return False

def test_rate_limiting_info():
    """Test rate limiting information (without triggering limits)."""
    print("\n" + "=" * 60)
    print("TEST 5: Rate Limiting Info")
    print("=" * 60)
    
    try:
        client = LLMClient()
        
        print(f"Max requests per minute: {client.MAX_REQUESTS_PER_MINUTE}")
        print(f"Max requests per day: {client.MAX_REQUESTS_PER_DAY}")
        print(f"Current requests this minute: {LLMClient.requests_this_minute}")
        print(f"Current requests today: {LLMClient.requests_today}")
        
        # Make a small request to test counter increment
        initial_count = LLMClient.requests_today
        client.generate_text("Hi", max_tokens=10)
        final_count = LLMClient.requests_today
        
        if final_count > initial_count:
            print("✓ Rate limiting counters working correctly")
            return True
        else:
            print("✗ Rate limiting counters not incrementing")
            return False
            
    except Exception as e:
        print(f"✗ Rate limiting test FAILED: {str(e)}")
        return False

def test_response_validation():
    """Test response validation functionality."""
    print("\n" + "=" * 60)
    print("TEST 6: Response Validation")
    print("=" * 60)
    
    client = LLMClient()
    
    # Test valid response
    valid_response = "This is a valid response."
    if client.validate_response(valid_response):
        print("✓ Valid response validation PASSED")
    else:
        print("✗ Valid response validation FAILED")
        return False
    
    # Test empty response
    if not client.validate_response(""):
        print("✓ Empty response validation PASSED")
    else:
        print("✗ Empty response validation FAILED")
        return False
    
    # Test None response
    if not client.validate_response(None):
        print("✓ None response validation PASSED")
    else:
        print("✗ None response validation FAILED")
        return False
    
    print("✓ Response validation test PASSED")
    return True

def test_throughput_optimization():
    """Test default throughput optimization."""
    print("\n" + "=" * 60)
    print("TEST 7: Throughput Optimization")
    print("=" * 60)
    
    try:
        client = LLMClient()
        
        # Test without specifying provider options (should default to throughput)
        prompt = "What is 2+2?"
        print(f"Prompt: {prompt}")
        
        response = client.generate_text(prompt, max_tokens=20, temperature=0.1)
        print(f"Response: {response}")
        
        if response:
            print("✓ Throughput optimization test PASSED")
            return True
        else:
            print("✗ Throughput optimization test FAILED - No response")
            return False
            
    except Exception as e:
        print(f"✗ Throughput optimization test FAILED: {str(e)}")
        return False

def test_uml_specific_prompt():
    """Test with a UML-specific prompt to verify domain expertise."""
    print("\n" + "=" * 60)
    print("TEST 8: UML-Specific Prompt")
    print("=" * 60)
    
    try:
        client = LLMClient()
        
        prompt = """
        Create a simple PlantUML use case diagram for a library management system.
        Include actors: Librarian, Student
        Include use cases: Borrow Book, Return Book, Search Catalog
        """
        
        print(f"UML Prompt: {prompt.strip()}")
        
        response = client.generate_text(prompt, max_tokens=500, temperature=0.1)
        print(f"Response: {response}")
        
        # Check if response contains UML-related keywords
        uml_keywords = ["@startuml", "@enduml", "actor", "usecase", "-->"]
        found_keywords = [kw for kw in uml_keywords if kw.lower() in response.lower()]
        
        print(f"Found UML keywords: {found_keywords}")
        
        if len(found_keywords) >= 2:
            print("✓ UML-specific prompt test PASSED")
            return True
        else:
            print("✗ UML-specific prompt test FAILED - Not enough UML content")
            return False
            
    except Exception as e:
        print(f"✗ UML-specific prompt test FAILED: {str(e)}")
        return False

def check_environment():
    """Check if environment is properly set up."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ .env file found")
    else:
        print("✗ .env file not found")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print("✓ OPENROUTER_API_KEY found in environment")
        print(f"  Key starts with: {api_key[:10]}...")
    else:
        print("✗ OPENROUTER_API_KEY not found in environment")
        return False
    
    return True

def main():
    """Run all tests."""
    print("LLMClient Test Suite")
    print("Testing OpenRouter API integration with meta-llama/llama-3.3-70b-instruct:free model")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please ensure:")
        print("1. .env file exists in the project root")
        print("2. OPENROUTER_API_KEY is set in the .env file")
        return
    
    # Run all tests
    tests = [
        test_basic_functionality,
        test_provider_order,
        test_fallback_options,
        test_parameter_validation,
        test_rate_limiting_info,
        test_response_validation,
        test_throughput_optimization,
        test_uml_specific_prompt
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Small delay between tests to respect rate limits
        except Exception as e:
            print(f"✗ Test {test.__name__} encountered an error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests PASSED! LLMClient is working correctly.")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. Some issues may need attention.")
    else:
        print("❌ Many tests failed. Please check your configuration and API key.")

if __name__ == "__main__":
    main() 