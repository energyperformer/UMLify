import os
import requests
from dotenv import load_dotenv
import logging
import json
import time

logger = logging.getLogger(__name__)

class LLMClient:
    # Class-level rate limiting variables
    last_request_time = 0
    requests_this_minute = 0
    requests_today = 0
    last_day_reset = 0
    
    # OpenRouter free tier rate limits
    MAX_REQUESTS_PER_MINUTE = 20  # OpenRouter limit for free models
    MAX_REQUESTS_PER_DAY = 50     # OpenRouter limit for free tier
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_retries = 2  # Reduced from 3 to 2
        self.retry_delay = 2  # seconds
        
        # Instance-level request counter for this run
        self.total_requests_this_run = 0
        
        # Initialize daily counter if needed
        if LLMClient.last_day_reset == 0:
            LLMClient.last_day_reset = int(time.time() / 86400)  # Current day
            logger.info(f"Rate limit protection initialized: {self.MAX_REQUESTS_PER_MINUTE} RPM, {self.MAX_REQUESTS_PER_DAY} RPD")
        
    def _make_api_request(self, prompt: str, max_tokens: int, temperature: float, provider_order=None, allow_fallbacks=None) -> str:
        """Make API request with retries and proper error handling. Supports provider order and fallback options."""
        for attempt in range(self.max_retries):
            try:
                # Check and enforce rate limits
                self._enforce_rate_limits()
                
                # Add delay between retries with exponential backoff
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Attempt {attempt+1}/{self.max_retries}: Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                
                # Clean the prompt to remove any problematic characters
                clean_prompt = prompt.strip()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "UMLify"
                }
                payload = {
                    "model": "meta-llama/llama-3.3-70b-instruct:free",  # Llama-3.3-70B-Instruct
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a UML expert. Generate detailed PlantUML diagrams that follow best practices."
                        },
                        {
                            "role": "user",
                            "content": clean_prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                # Add provider order/fallbacks if specified, or default to throughput optimization
                if provider_order or allow_fallbacks is not None:
                    payload["provider"] = {}
                    if provider_order:
                        payload["provider"]["order"] = provider_order
                    if allow_fallbacks is not None:
                        payload["provider"]["allow_fallbacks"] = allow_fallbacks
                else:
                    # Default to throughput optimization when no specific provider settings
                    payload["provider"] = {"sort": "throughput"}
                response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=60)
                if response.status_code != 200:
                    logger.error(f"OpenRouter API returned status {response.status_code}: {response.text}")
                    raise ValueError(f"OpenRouter API error: {response.status_code} {response.text}")
                completion = response.json()
                logger.debug(f"Raw API Response: {completion}")
                # Validate response structure
                if not completion:
                    raise ValueError("Empty completion object")
                if "choices" not in completion:
                    raise ValueError("No choices in completion")
                if not completion["choices"]:
                    raise ValueError("Empty choices list")
                if "message" not in completion["choices"][0]:
                    raise ValueError("No message in first choice")
                if "content" not in completion["choices"][0]["message"]:
                    raise ValueError("No content in message")
                content = completion["choices"][0]["message"]["content"]
                if not content or not isinstance(content, str):
                    raise ValueError("Invalid content type or empty content")
                # Clean the response content
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                logger.debug(f"API Response Content: {content[:200]}...")
                # Increment request counters after successful request
                self._increment_request_counters()
                return content
            except Exception as e:
                error_msg = str(e)
                logger.error(f"API request attempt {attempt + 1} failed: {error_msg}")
                # Check for rate limit errors specifically
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    rate_limit_delay = self.retry_delay * (4 ** attempt)  # More aggressive backoff for rate limits
                    logger.warning(f"Rate limit detected! Waiting {rate_limit_delay} seconds before retry...")
                    time.sleep(rate_limit_delay)
                elif attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError(f"All API request attempts failed: {error_msg}")

    def _enforce_rate_limits(self):
        """Enforce OpenRouter rate limits to prevent 429 errors"""
        current_time = time.time()
        current_day = int(current_time / 86400)  # Convert to days
        # Reset daily counter if it's a new day
        if current_day > self.last_day_reset:
            logger.info("New day detected, resetting daily request counter")
            LLMClient.requests_today = 0
            LLMClient.last_day_reset = current_day
        # Check daily limit - this is critical for free tier
        if LLMClient.requests_today >= self.MAX_REQUESTS_PER_DAY:
            wait_seconds_until_tomorrow = ((current_day + 1) * 86400) - current_time
            hours_remaining = wait_seconds_until_tomorrow / 3600
            logger.warning(f"Daily request limit of {self.MAX_REQUESTS_PER_DAY} reached. Need to wait until tomorrow.")
            raise ValueError(f"Daily request limit reached. Please try again in {hours_remaining:.1f} hours.")
        # Check per-minute limit
        time_since_last_minute = current_time - LLMClient.last_request_time
        # Reset counter if more than a minute has passed
        if time_since_last_minute > 60:
            LLMClient.requests_this_minute = 0
            LLMClient.last_request_time = current_time
        # If we've hit our limit, wait until next minute
        if LLMClient.requests_this_minute >= self.MAX_REQUESTS_PER_MINUTE:
            wait_time = 65 - time_since_last_minute  # 65 to be safe
            if wait_time > 0:
                logger.info(f"Rate limit protection: Waiting {wait_time:.2f}s to respect per-minute limits...")
                time.sleep(wait_time)
                # Reset after waiting
                LLMClient.requests_this_minute = 0
                LLMClient.last_request_time = time.time()

    def _increment_request_counters(self):
        """Increment request counters after a successful request"""
        LLMClient.requests_this_minute += 1
        LLMClient.requests_today += 1
        self.total_requests_this_run += 1  # Increment instance counter
        # Log request count status
        logger.info(f"Request count: {LLMClient.requests_this_minute}/{self.MAX_REQUESTS_PER_MINUTE} per minute, {LLMClient.requests_today}/{self.MAX_REQUESTS_PER_DAY} per day, {self.total_requests_this_run} this run")
        LLMClient.last_request_time = time.time()
        
    def generate_text(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2, provider_order=None, allow_fallbacks=None) -> str:
        """
        Generate text using OpenRouter API.
        Supports provider order and fallback options.
        
        Args:
            prompt (str): The input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.0 to 1.0)
            provider_order (list[str], optional): List of provider slugs to prioritize
            allow_fallbacks (bool, optional): Whether to allow fallback to other providers
        
        Returns:
            str: Generated text response
        
        Raises:
            ValueError: If API response is invalid or empty
        """
        try:
            # Validate input parameters
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Invalid prompt: must be a non-empty string")
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            # Make API request with retries
            content = self._make_api_request(prompt, max_tokens, temperature, provider_order, allow_fallbacks)
            # Validate response content
            if not self.validate_response(content):
                raise ValueError("Generated content failed validation")
            return content
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            raise ValueError(f"Failed to generate text: {str(e)}")

    def validate_response(self, response: str) -> bool:
        """
        Validate the response from the API with more permissive rules.
        
        Args:
            response (str): The API response to validate
        
        Returns:
            bool: True if response is valid, False otherwise
        """
        # Basic validation - just check if we have a string response
        if not response or not isinstance(response, str):
            logger.error("Invalid response type or empty response")
            return False
        # Accept any non-empty response, regardless of length
        # Short answers like "4" are valid for some queries
        if len(response.strip()) > 0:
            return True
        logger.error("Response is completely empty")
        return False
    
    def get_total_requests_this_run(self) -> int:
        """
        Get the total number of successful LLM requests made during this run.
        
        Returns:
            int: Total number of requests made by this LLMClient instance
        """
        return self.total_requests_this_run
    
    def get_request_summary(self) -> dict:
        """
        Get a summary of request counts for this run and overall.
        
        Returns:
            dict: Summary containing this run, daily, and per-minute counts
        """
        return {
            "this_run": self.total_requests_this_run,
            "today_total": LLMClient.requests_today,
            "current_minute": LLMClient.requests_this_minute,
            "daily_limit": self.MAX_REQUESTS_PER_DAY,
            "minute_limit": self.MAX_REQUESTS_PER_MINUTE,
            "daily_remaining": max(0, self.MAX_REQUESTS_PER_DAY - LLMClient.requests_today),
            "minute_remaining": max(0, self.MAX_REQUESTS_PER_MINUTE - LLMClient.requests_this_minute)
        } 