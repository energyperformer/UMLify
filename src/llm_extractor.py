import json
import logging
from typing import Dict, Any, List
from llm_client import OpenRouterClient

logger = logging.getLogger(__name__)

class LLMExtractor:
    def __init__(self):
        self.llm_client = OpenRouterClient()
        
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract actors and use cases from text using LLM.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing actors and use cases
        """
        prompt = f"""Analyze this text and extract actors and use cases.
Format the output as a JSON object with two arrays: 'actors' and 'use_cases'.

Rules:
1. Identify all actors (users, systems, external entities)
2. Identify all use cases (actions, functionalities)
3. For each use case, specify:
   - Name
   - Description
   - Associated actors
   - Preconditions
   - Postconditions
4. For each actor, specify:
   - Name
   - Type (user, system, external)
   - Description

Text to analyze:
{text}

Remember:
- Be precise in identifying actors and use cases
- Maintain proper relationships between actors and use cases
- Include all relevant details
- Format output as valid JSON"""

        try:
            logger.info("Sending text to LLM for entity extraction...")
            response = self.llm_client.generate_text(prompt)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
                
            json_str = response[json_start:json_end]
            entities = json.loads(json_str)
            
            # Validate structure
            if not isinstance(entities, dict) or 'actors' not in entities or 'use_cases' not in entities:
                raise ValueError("Invalid entity structure in response")
                
            logger.info(f"Extracted {len(entities['actors'])} actors and {len(entities['use_cases'])} use cases")
            return entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            raise 