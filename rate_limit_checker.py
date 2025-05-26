#!/usr/bin/env python3
"""
OpenRouter API Rate Limit Checker
Checks current rate limit status for OpenRouter API keys.
"""

import os
import sys
import requests
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

class OpenRouterRateLimitChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # OpenRouter free tier limits
        self.MAX_REQUESTS_PER_MINUTE = 20
        self.MAX_REQUESTS_PER_DAY = 50
        
    def check_rate_limits(self) -> dict:
        """
        Check current rate limit status by making a minimal API request.
        Returns detailed rate limit information.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "UMLify-RateChecker"
        }
        
        # Minimal payload to check rate limits
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": "Hi"
                }
            ],
            "max_tokens": 1,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)
            
            # Extract rate limit info from headers
            rate_limit_info = self._extract_rate_limit_headers(response.headers)
            
            # Add response status
            rate_limit_info['api_response'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'error': None if response.status_code == 200 else response.text
            }
            
            return rate_limit_info
            
        except requests.exceptions.RequestException as e:
            return {
                'error': f"Request failed: {str(e)}",
                'api_response': {
                    'status_code': None,
                    'success': False,
                    'error': str(e)
                }
            }
    
    def _extract_rate_limit_headers(self, headers) -> dict:
        """Extract rate limit information from response headers."""
        current_time = datetime.now()
        
        # Common rate limit header names (different APIs use different conventions)
        rate_limit_headers = {
            'requests_remaining': headers.get('x-ratelimit-remaining'),
            'requests_limit': headers.get('x-ratelimit-limit'),
            'reset_time': headers.get('x-ratelimit-reset'),
            'retry_after': headers.get('retry-after'),
            'requests_used': headers.get('x-ratelimit-used'),
        }
        
        # Try alternative header names
        if not rate_limit_headers['requests_remaining']:
            rate_limit_headers['requests_remaining'] = headers.get('x-ratelimit-requests-remaining')
        
        if not rate_limit_headers['requests_limit']:
            rate_limit_headers['requests_limit'] = headers.get('x-ratelimit-requests-limit')
        
        # Calculate time-based information
        result = {
            'timestamp': current_time.isoformat(),
            'headers_found': {},
            'estimated_limits': {
                'max_per_minute': self.MAX_REQUESTS_PER_MINUTE,
                'max_per_day': self.MAX_REQUESTS_PER_DAY
            },
            'time_info': {
                'current_minute': current_time.strftime('%H:%M'),
                'seconds_until_next_minute': 60 - current_time.second,
                'hours_until_midnight': 24 - current_time.hour,
                'minutes_until_midnight': (24 - current_time.hour) * 60 - current_time.minute
            }
        }
        
        # Add found headers
        for key, value in rate_limit_headers.items():
            if value is not None:
                result['headers_found'][key] = value
        
        # Parse reset time if available
        if rate_limit_headers['reset_time']:
            try:
                reset_timestamp = int(rate_limit_headers['reset_time'])
                reset_time = datetime.fromtimestamp(reset_timestamp)
                result['reset_info'] = {
                    'reset_time': reset_time.isoformat(),
                    'seconds_until_reset': max(0, int((reset_time - current_time).total_seconds())),
                    'minutes_until_reset': max(0, int((reset_time - current_time).total_seconds() / 60))
                }
            except (ValueError, TypeError):
                result['reset_info'] = {'error': 'Could not parse reset time'}
        
        # Calculate estimated usage if we have the data
        if rate_limit_headers['requests_remaining'] and rate_limit_headers['requests_limit']:
            try:
                remaining = int(rate_limit_headers['requests_remaining'])
                limit = int(rate_limit_headers['requests_limit'])
                used = limit - remaining
                
                result['usage_stats'] = {
                    'requests_used': used,
                    'requests_remaining': remaining,
                    'requests_limit': limit,
                    'usage_percentage': round((used / limit) * 100, 2) if limit > 0 else 0
                }
            except (ValueError, TypeError):
                pass
        
        return result
    
    def get_detailed_status(self) -> dict:
        """Get comprehensive rate limit status with recommendations."""
        rate_info = self.check_rate_limits()
        
        # Add recommendations based on current status
        recommendations = []
        
        if rate_info.get('api_response', {}).get('status_code') == 429:
            recommendations.append("⚠️ Rate limit exceeded! Wait before making more requests.")
        elif rate_info.get('api_response', {}).get('success'):
            recommendations.append("✅ API key is working correctly.")
        
        # Add usage recommendations
        usage_stats = rate_info.get('usage_stats', {})
        if usage_stats:
            usage_pct = usage_stats.get('usage_percentage', 0)
            if usage_pct > 90:
                recommendations.append("🔴 Very high usage! Consider slowing down requests.")
            elif usage_pct > 70:
                recommendations.append("🟡 High usage. Monitor your request rate.")
            elif usage_pct < 30:
                recommendations.append("🟢 Low usage. You have plenty of requests available.")
        
        # Add timing recommendations
        time_info = rate_info.get('time_info', {})
        if time_info.get('seconds_until_next_minute', 0) < 10:
            recommendations.append("⏰ New minute starting soon - rate limits may reset.")
        
        if time_info.get('hours_until_midnight', 0) < 2:
            recommendations.append("🌙 Daily limits reset at midnight.")
        
        rate_info['recommendations'] = recommendations
        return rate_info

def format_status_report(status: dict) -> str:
    """Format the status information into a readable report."""
    report = []
    report.append("=" * 60)
    report.append("🔍 OPENROUTER API RATE LIMIT STATUS")
    report.append("=" * 60)
    
    # Timestamp
    report.append(f"📅 Checked at: {status.get('timestamp', 'Unknown')}")
    
    # API Response Status
    api_response = status.get('api_response', {})
    if api_response.get('success'):
        report.append("✅ API Status: Working")
    else:
        report.append(f"❌ API Status: Error (Code: {api_response.get('status_code', 'Unknown')})")
        if api_response.get('error'):
            report.append(f"   Error: {api_response['error'][:100]}...")
    
    report.append("")
    
    # Usage Statistics
    usage_stats = status.get('usage_stats', {})
    if usage_stats:
        report.append("📊 CURRENT USAGE:")
        report.append(f"   Requests Used: {usage_stats.get('requests_used', 'Unknown')}")
        report.append(f"   Requests Remaining: {usage_stats.get('requests_remaining', 'Unknown')}")
        report.append(f"   Total Limit: {usage_stats.get('requests_limit', 'Unknown')}")
        report.append(f"   Usage: {usage_stats.get('usage_percentage', 0)}%")
    else:
        report.append("📊 ESTIMATED LIMITS:")
        estimated = status.get('estimated_limits', {})
        report.append(f"   Max per minute: {estimated.get('max_per_minute', 'Unknown')}")
        report.append(f"   Max per day: {estimated.get('max_per_day', 'Unknown')}")
    
    report.append("")
    
    # Time Information
    time_info = status.get('time_info', {})
    if time_info:
        report.append("⏰ TIME UNTIL RESET:")
        report.append(f"   Next minute: {time_info.get('seconds_until_next_minute', 0)} seconds")
        report.append(f"   Midnight (daily reset): {time_info.get('hours_until_midnight', 0)} hours, {time_info.get('minutes_until_midnight', 0) % 60} minutes")
    
    # Reset Information
    reset_info = status.get('reset_info', {})
    if reset_info and 'error' not in reset_info:
        report.append(f"   Rate limit reset: {reset_info.get('minutes_until_reset', 0)} minutes")
    
    report.append("")
    
    # Headers Found
    headers_found = status.get('headers_found', {})
    if headers_found:
        report.append("🔍 RATE LIMIT HEADERS FOUND:")
        for key, value in headers_found.items():
            report.append(f"   {key}: {value}")
        report.append("")
    
    # Recommendations
    recommendations = status.get('recommendations', [])
    if recommendations:
        report.append("💡 RECOMMENDATIONS:")
        for rec in recommendations:
            report.append(f"   {rec}")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Main function to run the rate limit checker."""
    print("OpenRouter API Rate Limit Checker")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from command line argument or environment
    api_key = None
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print(f"Using API key from command line argument")
    else:
        # Try to get from environment variables
        possible_keys = [
            "OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY_FALLBACK",
            "OPENROUTER_API_KEY_SRS",
            "OPENROUTER_API_KEY_EXTRACT",
            "OPENROUTER_API_KEY_PLANTUML"
        ]
        
        for key_name in possible_keys:
            api_key = os.getenv(key_name)
            if api_key:
                print(f"Using API key from environment variable: {key_name}")
                break
    
    if not api_key:
        print("❌ No API key found!")
        print("\nUsage:")
        print("  python rate_limit_checker.py <your-api-key>")
        print("  OR set one of these environment variables:")
        for key_name in possible_keys:
            print(f"    {key_name}")
        return
    
    # Mask the API key for display
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
    print(f"API Key: {masked_key}")
    print()
    
    # Check rate limits
    print("🔍 Checking rate limits...")
    checker = OpenRouterRateLimitChecker(api_key)
    
    try:
        status = checker.get_detailed_status()
        report = format_status_report(status)
        print(report)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rate_limit_report_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error checking rate limits: {str(e)}")

if __name__ == "__main__":
    main() 