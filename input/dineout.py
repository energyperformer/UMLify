import PyPDF2
import re
import json
import os
from collections import defaultdict

def extract_sections_from_pdf(pdf_path):
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file '{pdf_path}' does not exist.")
        return {}
    
    # Open and read the PDF
    with open(pdf_path, 'rb') as file:
        try:
            reader = PyPDF2.PdfReader(file)
            if len(reader.pages) == 0:
                print("WARNING: PDF has no pages.")
                return {}
            
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"ERROR reading PDF: {str(e)}")
            return {}
    
    # Split text into lines and clean up
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        print("WARNING: No text content found in PDF.")
        return {}
    
    # Debug info
    print(f"Successfully extracted {len(lines)} lines from PDF")
    
    # Initialize variables
    sections = defaultdict(list)
    current_section = "General"
    
    # Look for common section patterns in technical documents
    section_patterns = [
        r'^[A-Z][A-Z\s]+:?$', # ALL CAPS
        r'^[0-9]+\.[0-9]*\s+[A-Z]', # Numbered sections like "1.2 SECTION NAME"
        r'^[IVX]+\.\s+[A-Z]', # Roman numerals like "IV. SECTION NAME"
        r'^[A-Za-z\s]+:$', # Section name with colon
        r'^[0-9]+\.\s+[A-Za-z\s]+$' # Numbered section names
    ]
    
    # Process each line
    for i, line in enumerate(lines):
        # Check if line might be a section header
        is_section_header = False
        for pattern in section_patterns:
            if re.match(pattern, line):
                is_section_header = True
                current_section = line.rstrip(':').strip()
                break
        
        # If not a section header, add to current section
        if not is_section_header:
            # Simple heuristic: if line is short and next line is longer, it might be a header
            if i < len(lines) - 1 and len(line) < 50 and len(lines[i+1]) > 50:
                current_section = line.strip()
            else:
                sections[current_section].append(line)
    
    # If we couldn't identify any sections, create a main section with all content
    if len(sections) == 0 or (len(sections) == 1 and "General" in sections):
        print("WARNING: No clear sections identified. Creating single section with all content.")
        if "General" in sections:
            all_content = sections["General"]
        else:
            all_content = lines
            
        # Try to identify title or main sections from content
        if all_content:
            # Use first line as document title if it's short
            title = all_content[0] if len(all_content[0]) < 100 else "Document Content"
            sections = {"Document Title": [title], "Main Content": all_content[1:] if title == all_content[0] else all_content}
    
    return sections

def clean_section_content(content):
    # Remove empty lines and clean up the content
    return [line for line in content if line.strip()]

def main():
    pdf_path = 'input/dineout.pdf'
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Extract sections from PDF
    print(f"Extracting content from: {pdf_path}")
    sections = extract_sections_from_pdf(pdf_path)
    
    # Clean up section contents
    cleaned_sections = {
        section: clean_section_content(content)
        for section, content in sections.items()
    }
    
    # Convert to JSON format
    output_data = {
        "document_sections": cleaned_sections
    }
    
    # Write to JSON file
    output_path = 'output/dineout_info.json'
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, indent=4, ensure_ascii=False)
    
    print(f"Successfully extracted {len(cleaned_sections)} sections from the PDF")
    for section, content in cleaned_sections.items():
        print(f" - '{section}': {len(content)} lines")
    print(f"JSON file has been generated at: {output_path}")

    # Create a text version of the PDF for easier debugging
    text_path = 'output/dineout_text.txt'
    with open(text_path, 'w', encoding='utf-8') as text_file:
        for section, content in cleaned_sections.items():
            text_file.write(f"=== {section} ===\n")
            text_file.write("\n".join(content))
            text_file.write("\n\n")
    print(f"Text version saved at: {text_path}")

if __name__ == "__main__":
    main() 