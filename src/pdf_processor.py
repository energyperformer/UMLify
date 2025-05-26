import pdfplumber
import json
from typing import Dict, Any, List, Optional
import logging
import re
from dataclasses import dataclass
import os

logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed logging
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
    def __init__(self):
        self.text = ""
        self.metadata = {}
        self.sections = []
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text. Less aggressive: only remove empty lines and lines that are just numbers."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0
        for line in lines:
            stripped = line.strip()
            # Remove empty lines or lines that are just numbers
            if not stripped or stripped.isdigit():
                removed_count += 1
                continue
            cleaned_lines.append(line)
        logger.debug(f"_clean_text: Removed {removed_count} lines out of {len(lines)} total lines.")
        return '\n'.join(cleaned_lines)
        
    def _extract_metadata(self, pdf: pdfplumber.PDF) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}
        if pdf.metadata:
            metadata = {
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
                'creator': pdf.metadata.get('Creator', ''),
                'producer': pdf.metadata.get('Producer', ''),
                'subject': pdf.metadata.get('Subject', ''),
                'keywords': pdf.metadata.get('Keywords', ''),
                'page_count': len(pdf.pages)
            }
        return metadata
        
    def _extract_all_text(self, pdf: pdfplumber.PDF) -> str:
        """Extract text from all pages with page markers."""
        all_text = []
        total_pages = len(pdf.pages)
        logger.info(f"Total pages in PDF: {total_pages}")
        
        for i, page in enumerate(pdf.pages, 1):
            try:
                logger.debug(f"Processing page {i}/{total_pages}")
                
                # Try different extraction methods
                page_text = page.extract_text()
                if not page_text:
                    # Try extracting text from words
                    words = page.extract_words()
                    if words:
                        page_text = ' '.join(word['text'] for word in words)
                
                if page_text:
                    # Add page marker
                    all_text.append(f"\n--- Page {i} ---\n")
                    all_text.append(page_text)
                    logger.debug(f"Extracted {len(page_text)} characters from page {i}")
                else:
                    logger.warning(f"No text extracted from page {i}")
                    
            except Exception as e:
                logger.error(f"Error extracting text from page {i}: {str(e)}")
                continue
                
        extracted_text = '\n'.join(all_text)
        logger.info(f"Total extracted text length: {len(extracted_text)} characters")
        return extracted_text

    def extract_text(self, pdf_path: str) -> None:
        """
        Extract text and metadata from PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF file is invalid or empty
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            logger.info(f"Processing PDF: {pdf_path}")
            logger.debug(f"PDF file size: {os.path.getsize(pdf_path)} bytes")
            
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF file is empty")
                    
                # Extract metadata
                self.metadata = self._extract_metadata(pdf)
                logger.debug(f"Extracted metadata: {self.metadata}")
                
                # Extract text from all pages
                raw_text = self._extract_all_text(pdf)
                if not raw_text:
                    raise ValueError("No text could be extracted from PDF")
                    
                # Clean and normalize text
                self.text = self._clean_text(raw_text)
                if not self.text:
                    raise ValueError("No text remained after cleaning")
                    
                logger.info(f"Successfully extracted {len(self.text)} characters of text")
                
                # Print first 500 characters for debugging
                logger.debug(f"First 500 characters of extracted text:\n{self.text[:500]}")
                
                # Save the extracted text to a file for debugging
                debug_file = os.path.join(os.path.dirname(pdf_path), 'debug_extracted_text.txt')
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(self.text)
                logger.debug(f"Saved extracted text to {debug_file}")
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
            
    def get_text(self) -> str:
        """Get the extracted text."""
        return self.text
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get the extracted metadata."""
        return self.metadata

    def save_to_json(self, output_path: str):
        """Save extracted text to JSON file."""
        data = {
            'text': self.text,
            'metadata': self.metadata
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_path}") 