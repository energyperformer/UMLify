# UMLify

A tool to automatically generate UML diagrams from SRS documents using LLM and OpenRouter.

## Features

- PDF text extraction and analysis
- LLM-powered entity extraction
- Neo4j graph database integration
- PlantUML diagram generation
- OpenRouter API integration

## Prerequisites

- Python 3.8+
- Neo4j Database
- OpenRouter API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/umlify.git
cd umlify
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with the following variables:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# PlantUML Configuration
PLANTUML_SERVER_URL=http://www.plantuml.com/plantuml/img/
```

## Usage

1. Place your SRS PDF file in the project root as `requirements.pdf`

2. Run the main script:
```bash
python src/main.py
```
Also make sure in main.py file add ur password for neo4j dbms

3. The script will:
   - Extract text from the PDF
   - Analyze the content using LLM
   - Store entities in Neo4j
   - Generate a PlantUML diagram
   - Save the diagram as `diagram.png`

## Output Files

- `extracted_text.json`: Extracted and analyzed text from PDF
- `plantuml.txt`: Generated PlantUML code
- `diagram.png`: Generated UML diagram
- `neo4j_dump.json`: Exported Neo4j data

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys and passwords secure
- Use environment variables for all sensitive data

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



