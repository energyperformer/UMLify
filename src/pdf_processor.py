import pdfplumber
import json
from typing import Dict, Any, List
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from llm_client import OpenRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Section:
    title: str
    content: str
    type: str
    page: int
    level: int = 0
    subsections: List['Section'] = None
    metadata: Dict[str, Any] = None

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.llm_client = OpenRouterClient()
        self.sections = []
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)
        return text.strip()
        
    def _extract_metadata(self, pdf) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {
            'title': pdf.metadata.get('Title', ''),
            'author': pdf.metadata.get('Author', ''),
            'creation_date': pdf.metadata.get('CreationDate', ''),
            'page_count': len(pdf.pages),
            'processed_date': datetime.now().isoformat()
        }
        logger.info(f"Extracted metadata: {metadata}")
        return metadata
        
    def _extract_all_text(self, pdf) -> str:
        """Extract all text from PDF with page markers."""
        full_text = []
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                # Add page marker
                full_text.append(f"\n--- Page {page_num} ---\n")
                full_text.append(text)
        return "\n".join(full_text)

    def _summarize_text(self, text: str) -> Dict[str, Any]:
        """Summarize text using LLM to extract key information."""
        prompt = f"""Analyze this SRS document and create a structured summary focusing on system requirements and functionality.

Rules:
1. Extract key information about:
   - System overview and purpose
   - Main actors/users
   - Core functionalities
   - System requirements
   - Use cases
2. Format the output as a JSON object with the following structure:
{{
    "system_overview": "Brief description of the system",
    "actors": [
        {{
            "name": "actor name",
            "type": "user/system/external",
            "description": "actor description"
        }}
    ],
    "core_functionalities": [
        {{
            "name": "functionality name",
            "description": "detailed description"
        }}
    ],
    "requirements": [
        {{
            "type": "functional/non-functional",
            "description": "requirement description"
        }}
    ],
    "use_cases": [
        {{
            "name": "use case name",
            "actors": ["actor1", "actor2"],
            "description": "use case description",
            "preconditions": ["precondition1"],
            "postconditions": ["postcondition1"]
        }}
    ]
}}

Document Content:
{text}

Remember:
- Be precise and concise
- Focus on information relevant for UML generation
- Ensure valid JSON format
- Include all important actors and use cases"""

        try:
            logger.info("Sending text to LLM for summarization...")
            response = self.llm_client.generate_text(prompt, max_tokens=2000)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
                
            json_str = response[json_start:json_end]
            
            # Clean the JSON string
            json_str = re.sub(r'[\n\r\t]', '', json_str)  # Remove newlines and tabs
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # Remove comments
            
            try:
                summary = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {str(e)}")
                logger.error(f"JSON string: {json_str}")
                raise
                
            # Validate structure
            required_keys = ['system_overview', 'actors', 'core_functionalities', 'requirements', 'use_cases']
            if not all(key in summary for key in required_keys):
                raise ValueError(f"Missing required fields. Found: {list(summary.keys())}")
                
            logger.info(f"Successfully extracted summary with {len(summary['actors'])} actors and {len(summary['use_cases'])} use cases")
            return summary
            
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            raise
        
    def extract_text(self) -> Dict[str, Any]:
        """Extract text from PDF and create structured summary."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Extract metadata
                metadata = self._extract_metadata(pdf)
                logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                
                # Extract all text
                full_text = self._extract_all_text(pdf)
                logger.info(f"Extracted {len(full_text.split())} words")
                
                # Create structured summary
                summary = self._summarize_text(full_text)
                
                return {
                    "metadata": metadata,
                    "summary": summary
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
            
    def save_to_json(self, output_path: str):
        """Save extracted text to JSON file."""
        data = self.extract_text()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Total actors: {len(data['summary']['actors'])}")
        logger.info(f"Total use cases: {len(data['summary']['use_cases'])}") 