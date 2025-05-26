import os
import sys
# import logging # Removed
import json
from datetime import datetime
from llm_extractor import LLMExtractor
from plantuml_generator import PlantUMLGenerator
from neo4j_operations import Neo4jOperations
from dotenv import load_dotenv
import re # For sorting numbered directories
load_dotenv()
 
# Constants and configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
TEXT_FILE_PATH = os.getenv("TEXT_FILE_PATH", "input/dineout.txt")

# --- Function to determine the next output directory number ---
def get_next_output_dir_number(base_output_path="output") -> int:
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
        return 1
    
    existing_dirs = [d for d in os.listdir(base_output_path) 
                     if os.path.isdir(os.path.join(base_output_path, d)) and d.isdigit()]
    
    if not existing_dirs:
        return 1
    
    last_num = 0
    for dir_name in existing_dirs:
        try:
            num = int(dir_name)
            if num > last_num:
                last_num = num
        except ValueError:
            continue # Skip non-numeric directory names just in case
            
    return last_num + 1

# Create a timestamped and numbered output directory
NEXT_OUTPUT_NUM = get_next_output_dir_number()
# TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S") # Optional: include timestamp if preferred
OUTPUT_DIR = os.path.join("output", str(NEXT_OUTPUT_NUM))
# --- End of output directory modification ---

def setup_output_directories():
    """Setup output directories."""
    # OUTPUT_DIR is now globally defined and created if needed by get_next_output_dir_number
    # So, we just ensure the subdirectories are created here.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created base output directory: {OUTPUT_DIR}") # Replaced logger.info

    directories = {
        "json": os.path.join(OUTPUT_DIR, "json"),
        "textbook": os.path.join(OUTPUT_DIR, "textbook"),
        "plantuml": os.path.join(OUTPUT_DIR, "plantuml"),
        "diagrams": os.path.join(OUTPUT_DIR, "diagrams"),
        "neo4j": os.path.join(OUTPUT_DIR, "neo4j")
    }
    
    for name, directory_path in directories.items():
        os.makedirs(directory_path, exist_ok=True)
        print(f"Ensured {name} directory exists: {directory_path}") # Replaced logger.info
        
    return directories

def main():
    """Main workflow for UML diagram generation."""
    print(f"===== Starting UMLify Process =====") # Replaced logger.info
    print(f"Output will be in: {OUTPUT_DIR}") # Replaced logger.info
    neo4j_manager = None
    try:
        print("Step 0: Setting up output directories...") # Replaced logger.info
        directories = setup_output_directories()
        print("Step 1: Output directories setup complete.") # Replaced logger.info
        
        print("Attempting to initialize Neo4j connection...") # Replaced logger.info
        neo4j_connected = False
        try:
            neo4j_manager = Neo4jOperations(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
            neo4j_connected = True
            print("Successfully connected to Neo4j.") # Replaced logger.info
            
            # Clear the database after successful connection
            print("Clearing existing Neo4j database...") # Replaced logger.info
            neo4j_manager.clear_database() # This method should log its own success/failure
            print("Neo4j database cleared.") # Replaced logger.info
            
        except Exception as e:
            print(f"WARNING: Failed to connect to Neo4j or clear database: {str(e)}. Continuing without knowledge graph features.") # Replaced logger.warning
            neo4j_manager = None
            neo4j_connected = False # Ensure this is false if connection or clearing fails
        
        input_text_path = TEXT_FILE_PATH
        print(f"Reading input text from: {input_text_path}...") # Replaced logger.info
        try:
            with open(input_text_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
            print(f"Successfully read {len(extracted_text)} characters from {input_text_path}.") # Replaced logger.info
        except FileNotFoundError:
            print(f"ERROR: Input text file not found: {input_text_path}") # Replaced logger.error
            raise
        except Exception as e:
            print(f"ERROR: Error reading text file {input_text_path}: {str(e)}") # Replaced logger.error
            raise

        if not extracted_text:
            print(f"ERROR: No text could be read from {input_text_path}. Aborting.") # Replaced logger.error
            raise ValueError(f"No text could be read from {input_text_path}")
            
        input_text_json_path = os.path.join(directories["json"], "input_text.json")
        with open(input_text_json_path, 'w', encoding='utf-8') as f:
            json.dump({"text": extracted_text, "source_file": input_text_path}, f, indent=2, ensure_ascii=False)
        print(f"Saved raw input text to {input_text_json_path}") # Replaced logger.info
        
        print("Step 1: Generating standardized SRS...") 
        llm_extractor = LLMExtractor()
        
        # Configure LLM extractor to run more efficiently and avoid rate limits
        llm_extractor.min_request_interval = 5  # Increase time between requests to 5 seconds
        llm_extractor.max_retries = 5  # Increase max retries
        llm_extractor.llm_client.retry_delay = 5  # Increase base retry delay
        
        print("Configured LLM extractor with rate-limit protection")
        
        # Batch processing is useful but don't optimize prompts
        llm_extractor.batch_processing = True  # Keep batching for efficiency
        llm_extractor.optimize_prompts = False  # Don't optimize prompts as requested
        
        standardized_srs = llm_extractor.generate_standardized_srs(extracted_text)
        
        if not standardized_srs:
            print("ERROR: Failed to generate standardized SRS. Aborting.") 
            raise ValueError("Failed to generate standardized SRS")
        print("Step 2: Successfully generated standardized SRS.") 
        
        srs_path = os.path.join(directories["json"], "standardized_srs.json")
        with open(srs_path, "w", encoding="utf-8") as f:
            json.dump(standardized_srs, f, indent=2, ensure_ascii=False)
        print(f"Saved standardized SRS to {srs_path}") 
        
        print("Step 3: Extracting entities using LLM from standardized SRS...") 
        entities = llm_extractor.extract_entities(standardized_srs)
        
        if not entities:
            print("ERROR: Failed to extract entities using LLM. Aborting.") 
            raise ValueError("Failed to extract entities using LLM")
        print(f"Step 3: Successfully extracted {len(entities.get('actors',[]))} actors and {len(entities.get('use_cases',[]))} use cases. and {len(entities.get('relationships',[]))} relationships") # Replaced logger.info
        
        entities_path = os.path.join(directories["json"], "entities.json")
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
        print(f"Saved extracted entities to {entities_path}")
        
        if neo4j_connected and neo4j_manager:
            print("Step 4: Storing entities in Neo4j knowledge graph...") 
            if neo4j_manager.store_analysis_results(entities):
                print("Step 4: Successfully stored entities in Neo4j.") 
            else:
                print("WARNING: Step 4: Failed to store all entities in Neo4j. Check Neo4j logs for details.")
        else:
            print("Step 4: Skipping Neo4j storage (Neo4j not connected).") 
            
        print("Step 5: Generating PlantUML code...") # Replaced logger.info
        plantuml_generator = PlantUMLGenerator(neo4j_manager)
        
        # Configure rate limit protection for PlantUML generator
        plantuml_generator.min_request_interval = 5  # 5 seconds between requests
        print("Configured PlantUML generator with rate-limit protection")
        
        # Use template generation but don't optimize prompts
        plantuml_generator.optimize_for_free_tier = False  # Don't trim prompts
        
        plantuml_path = os.path.join(directories["plantuml"], "use_case_diagram.puml")

        # Generate PlantUML directly from entities.json
        # KG is only used for validation if available
        print("Generating PlantUML code directly from entities.json")
        plantuml_code = plantuml_generator.generate_plantuml(entities) 

        if plantuml_code:
            with open(plantuml_path, "w", encoding="utf-8") as f:
                f.write(plantuml_code)
            print(f"Step 5: Successfully generated and saved PlantUML code to {plantuml_path}.") # Replaced logger.info
        else:
            print("ERROR: Failed to generate PlantUML code. Aborting.") # Replaced logger.error
            raise ValueError("PlantUML code generation failed.")
        
        print("Step 6: Generating diagram from PlantUML code...") # Replaced logger.info
        if plantuml_generator.generate_diagram_from_plantuml(plantuml_path, directories["diagrams"]):
            final_diagram_path = os.path.join(directories["diagrams"], os.path.splitext(os.path.basename(plantuml_path))[0] + ".png")
            print(f"Step 6: Successfully generated diagram: {final_diagram_path}") # Replaced logger.info
        else:
            print("WARNING: Step 6: Failed to generate diagram image. Check PlantUML setup and logs.") # Replaced logger.warning
        
        if neo4j_connected and neo4j_manager:
            print("Step 7: Exporting Neo4j graph for visualization...") # Replaced logger.info
            project_name_for_export = os.path.basename(TEXT_FILE_PATH).split('.')[0]
            if neo4j_manager.export_graph(project_name_for_export):
                neo4j_export_dir = os.path.join(directories['neo4j'], project_name_for_export.lower().replace(' ', '_'))
                final_neo4j_export_path = os.path.join(neo4j_export_dir, "graph_export.json")
                print(f"Step 7: Successfully exported Neo4j graph to {final_neo4j_export_path}.") # Replaced logger.info
            else:
                print("WARNING: Step 7: Failed to export Neo4j graph. Check Neo4j logs.") # Replaced logger.warning
        else:
            print("Step 7: Skipping Neo4j export (Neo4j not connected).") # Replaced logger.info
            
        print("===== UMLify Process Completed Successfully =====") # Replaced logger.info
        final_diagram_path_to_print = os.path.join(directories["diagrams"], "use_case_diagram.png")
        if os.path.exists(final_diagram_path_to_print):
            print(f"Generated diagram: {final_diagram_path_to_print}")
        if os.path.exists(plantuml_path):
            print(f"Generated PlantUML code: {plantuml_path}")
            
        # Display LLM request summary
        print("\n===== LLM REQUEST SUMMARY =====")
        try:
            total_requests = llm_extractor.llm_client.get_total_requests_this_run()
            request_summary = llm_extractor.llm_client.get_request_summary()
            
            print(f"🔢 Total LLM calls this run: {total_requests}")
            print(f"📊 Daily usage: {request_summary['today_total']}/{request_summary['daily_limit']} ({request_summary['today_total']/request_summary['daily_limit']*100:.1f}%)")
            print(f"⏱️  Current minute: {request_summary['current_minute']}/{request_summary['minute_limit']}")
            print(f"🔋 Remaining today: {request_summary['daily_remaining']} requests")
            print(f"⚡ Remaining this minute: {request_summary['minute_remaining']} requests")
            print("=" * 32)
        except Exception as e:
            print(f"Could not retrieve LLM request summary: {str(e)}")
            
    except Exception as e:
        print(f"ERROR: Error in main workflow: {str(e)}") # Replaced logger.error
        print(f"AN ERROR OCCURRED: {str(e)}")
        print("===== UMLify Process Terminated with Errors =====") # Replaced logger.info
        
        # Display LLM request summary even on error
        print("\n===== LLM REQUEST SUMMARY (Error Case) =====")
        try:
            if 'llm_extractor' in locals():
                total_requests = llm_extractor.llm_client.get_total_requests_this_run()
                request_summary = llm_extractor.llm_client.get_request_summary()
                
                print(f"🔢 Total LLM calls this run: {total_requests}")
                print(f"📊 Daily usage: {request_summary['today_total']}/{request_summary['daily_limit']} ({request_summary['today_total']/request_summary['daily_limit']*100:.1f}%)")
                print(f"🔋 Remaining today: {request_summary['daily_remaining']} requests")
                print("=" * 32)
            else:
                print("LLM extractor not initialized yet")
        except Exception as summary_error:
            print(f"Could not retrieve LLM request summary: {str(summary_error)}")
    finally:
        if neo4j_manager:
            try:
                print("Closing Neo4j connection...") # Replaced logger.info
                neo4j_manager.close()
                print("Neo4j connection closed.") # Replaced logger.info
            except Exception as e:
                print(f"ERROR: Error closing Neo4j connection during finally block: {str(e)}") # Replaced logger.error

if __name__ == "__main__":
    main() 