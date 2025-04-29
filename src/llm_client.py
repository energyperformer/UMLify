import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """
        Generate text using OpenRouter API.
        
        Args:
            prompt (str): The input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.0 to 1.0)
            
        Returns:
            str: Generated text response
        """
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "UMLify"
                },
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise 