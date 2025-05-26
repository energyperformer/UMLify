# UMLify: Intelligent UML Diagram Generation

UMLify is a tool that generates accurate UML diagrams from Software Requirements Specification (SRS) documents using a hybrid approach that combines NLP techniques with Large Language Models (LLM) and Knowledge Graph verification.

## Overview

UMLify implements a robust methodology for UML diagram generation:

1. **PDF Processing** - Extract text from SRS PDF documents
2. **Standardized SRS Generation** - Summarize and standardize the SRS in a structured format
3. **Hybrid Entity Extraction** - Use both NLP and LLM to identify UML elements
4. **Knowledge Graph Construction** - Populate Neo4j with verified entities and relationships
5. **Contextual UML Generation** - Use textbook guidance and facts from the KG to generate accurate diagrams

## Hybrid Approach

UMLify uses a hybrid approach that combines:

- **Light NLP Techniques**: Dependency parsing and POS tagging to identify candidate entities
- **LLM Verification**: Confirming and classifying entity candidates 
- **Knowledge Graph**: Storing verified facts to prevent hallucination
- **RAG from Textbooks**: Using UML textbook guidelines as context for diagram generation

This approach mitigates issues with LLM-only solutions such as hallucination and inconsistency.

## Components

- `pdf_processor.py` - Extracts and processes text from PDF files
- `llm_extractor.py` - Hybrid approach for entity extraction using LLM 
- `nlp_extractor.py` - NLP-based techniques for candidate entity extraction
- `neo4j_operations.py` - Knowledge graph operations
- `textbook_utils.py` - Utility for extracting UML guidance from textbooks
- `plantuml_generator.py` - Generate PlantUML code and diagrams
- `main.py` - Main workflow orchestration

## Methodology Flow

1. **SRS PDF Extraction**
   - Extract text from PDF documents
   - Preprocess and clean the text

2. **SRS Standardization**
   - Summarize the document
   - Format according to IEEE SRS structure

3. **Entity Extraction (Hybrid Approach)**
   - Use NLP to extract candidate actors and use cases
   - Send candidates to LLM for verification and completion
   - Consult textbook guidance for validation

4. **Knowledge Graph Construction**
   - Store verified actors and use cases as nodes
   - Store relationships between entities
   - Create constraints and indexes

5. **UML Code Generation**
   - Retrieve facts from knowledge graph using Cypher queries
   - Apply UML best practices from textbooks
   - Generate syntactically correct PlantUML code

6. **Diagram Generation**
   - Convert PlantUML code to diagram images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/umlify.git
cd umlify
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

5. Set up Neo4j:
   - Install Neo4j or use a cloud instance
   - Create a database
   - Set environment variables

## Configuration

Create a `.env` file with the following variables:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# PlantUML Configuration
PLANTUML_SERVER_URL=http://www.plantuml.com/plantuml/img/
```

### OpenRouter Free Tier Rate Limits

UMLify uses OpenRouter's API with free tier rate limits:

- **Requests per Minute (RPM):** 20 requests per minute for free models
- **Requests per Day (RPD):** 50 requests per day on the free tier

The application has built-in rate limiting to avoid hitting these limits. For efficient usage with the free tier:

1. **Batch processing:** Process multiple entities in a single request where possible
2. **Plan your usage:** With only 50 requests per day, plan your diagrams carefully
3. **Retry handling:** The app includes automatic retry with exponential backoff

If you need to process larger projects and exceed these limits, consider upgrading to OpenRouter's paid tier (minimum 10 credits).

## Usage

Run the main script:

```bash
python src/main.py
```

By default, the script processes `input/requirements.pdf` and generates output in the `output` directory.

## Output Structure

Each run creates a numbered directory in `output/` containing:
- `json/` - JSON files with extracted data
- `plantuml/` - PlantUML code files
- `diagrams/` - Generated diagram images 
- `neo4j/` - Neo4j export files
- `textbook/` - Extracted textbook guidance
- `nlp/` - NLP extraction results

## Requirements

- Python 3.8+
- Neo4j
- OpenAI API key (via OpenRouter)
- PlantUML

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



