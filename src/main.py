import os
import json
import logging
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from graph_builder import GraphBuilder
from plantuml_generator import PlantUMLGenerator

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    graph_builder = None
    try:
        # Load environment variables
        logger.info("Loading environment variables...")
        load_dotenv()
        
        # Validate required environment variables
        required_vars = [
            'OPENROUTER_API_KEY',
            'NEO4J_URI',
            'NEO4J_USER',
            'NEO4J_PASSWORD'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Initialize processors
        pdf_processor = PDFProcessor("requirements.pdf")
        graph_builder = GraphBuilder(
            uri="bolt://127.0.0.1:7687",
            user="neo4j",
            password="YOUR_PASSWORD"

            # uri=os.getenv('NEO4J_URI'),
            # user=os.getenv('NEO4J_USER'),
            # password=os.getenv('NEO4J_PASSWORD')
        )
        plantuml_generator = PlantUMLGenerator()
        
        # 1. Process PDF and create summary
        logger.info("Step 1: Processing PDF and creating summary...")
        pdf_processor.save_to_json("extracted_text.json")
        logger.info("PDF processed and saved to extracted_text.json")
        
        # 2. Extract entities from summary
        logger.info("Step 2: Extracting entities from summary...")
        with open("extracted_text.json", "r") as f:
            data = json.load(f)
            summary = data["summary"]
            
        # Store actors
        for actor in summary["actors"]:
            graph_builder.store_actor(actor)
            
        # Store use cases
        for use_case in summary["use_cases"]:
            graph_builder.store_use_case(use_case)
            
        # Store requirements
        for requirement in summary["requirements"]:
            graph_builder.store_requirement(requirement)
            
        # Store core functionalities
        for functionality in summary["core_functionalities"]:
            graph_builder.store_functionality(functionality)
            
        # 3. Generate PlantUML
        logger.info("Step 3: Generating PlantUML...")
        graph_data = graph_builder.get_graph_data()
        uml_code = plantuml_generator.generate_uml(graph_data)
        
        # Validate and save UML code
        if plantuml_generator.validate_uml(uml_code):
            with open("plantuml.txt", "w") as f:
                f.write(uml_code)
            logger.info("PlantUML code saved to plantuml.txt")
            
            # Generate diagram
            logger.info("Generating diagram...")
            plantuml_generator.generate_diagram(uml_code, "diagram.png")
            logger.info("Diagram generated as diagram.png")
        else:
            logger.error("Invalid PlantUML syntax")
            
        # 4. Export Neo4j data
        logger.info("Step 4: Exporting Neo4j data...")
        with open("neo4j_dump.json", "w") as f:
            json.dump(graph_data, f, indent=2)
        logger.info("Neo4j data exported to neo4j_dump.json")
            
    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}")
        raise
    finally:
        if graph_builder:
            graph_builder.close()
            logger.info("Neo4j connection closed")
        logger.info("Workflow completed")

if __name__ == "__main__":
    main() 