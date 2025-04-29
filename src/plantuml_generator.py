import logging
from typing import Dict, Any, List
import plantuml
import tempfile
import os
from llm_client import OpenRouterClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class PlantUMLGenerator:
    def __init__(self):
        load_dotenv()
        self.llm_client = OpenRouterClient()
        self.plantuml_server = os.getenv('PLANTUML_SERVER_URL', 'http://www.plantuml.com/plantuml/img/')
        self.uml_template = """
@startuml
!theme plain
skinparam actorStyle awesome
skinparam usecase {{
    BackgroundColor<< Main >> YellowGreen
    BorderColor<< Main >> DarkGreen
    BackgroundColor<< Secondary >> LightBlue
    BorderColor<< Secondary >> DarkBlue
}}

title {title}

' Actors
{actors}

' Use Cases
rectangle "System" {{
{use_cases}
}}

' Relationships
{relationships}

@enduml
"""
        
    def generate_uml(self, graph_data: Dict[str, Any]) -> str:
        """Generate PlantUML code from graph data."""
        try:
            # Extract data
            actors = graph_data["actors"]
            use_cases = graph_data["use_cases"]
            relationships = graph_data["relationships"]
            
            # Generate actors section
            actors_section = "\n".join([
                f'actor "{actor["name"]}" as {self._format_name(actor["name"])}'
                for actor in actors
            ])
            
            # Generate use cases section
            use_cases_section = "\n".join([
                f'  usecase "{uc["name"]}" as {self._format_name(uc["name"])}'
                for uc in use_cases
            ])
            
            # Generate relationships section
            relationships_section = "\n".join([
                f'{self._format_name(rel["actor"])} --> {self._format_name(rel["use_case"])}'
                for rel in relationships
            ])
            
            # Generate initial UML code
            initial_uml = self.uml_template.format(
                title="System Use Case Diagram",
                actors=actors_section,
                use_cases=use_cases_section,
                relationships=relationships_section
            )
            
            # Use LLM to validate and correct the UML code
            corrected_uml = self._validate_and_correct_uml(initial_uml)
            
            logger.info("Successfully generated PlantUML code")
            return corrected_uml
            
        except Exception as e:
            logger.error(f"Error generating UML: {str(e)}")
            raise
            
    def _validate_and_correct_uml(self, uml_code: str) -> str:
        """Use LLM to validate and correct PlantUML code."""
        prompt = f"""Validate and correct this PlantUML code for a use case diagram. 
Fix any syntax errors and ensure it follows PlantUML best practices.

Rules:
1. Ensure proper @startuml and @enduml directives
2. Validate actor and use case definitions
3. Check relationship syntax
4. Ensure proper system boundary
5. Fix any formatting issues

Return ONLY the corrected PlantUML code, no explanations.

PlantUML Code:
{uml_code}"""

        try:
            response = self.llm_client.generate_text(prompt, max_tokens=2000)
            
            # Extract the PlantUML code from the response
            # Look for content between @startuml and @enduml
            start_idx = response.find("@startuml")
            end_idx = response.find("@enduml") + len("@enduml")
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("Invalid PlantUML code in LLM response")
                
            corrected_code = response[start_idx:end_idx]
            
            # Basic validation
            if not corrected_code.startswith("@startuml"):
                raise ValueError("Missing @startuml directive")
            if not corrected_code.endswith("@enduml"):
                raise ValueError("Missing @enduml directive")
                
            return corrected_code
            
        except Exception as e:
            logger.error(f"Error in UML validation: {str(e)}")
            raise
            
    def _format_name(self, name: str) -> str:
        """Format name for PlantUML identifier."""
        # Remove special characters and spaces
        formatted = ''.join(c for c in name if c.isalnum() or c == '_')
        # Ensure it starts with a letter
        if not formatted[0].isalpha():
            formatted = 'a' + formatted
        return formatted.lower()
        
    def validate_uml(self, uml_code: str) -> bool:
        """Validate PlantUML syntax."""
        try:
            # Basic validation
            if not uml_code.startswith("@startuml"):
                logger.error("Missing @startuml directive")
                return False
            if not uml_code.endswith("@enduml"):
                logger.error("Missing @enduml directive")
                return False
            if "actor" not in uml_code:
                logger.error("No actors defined")
                return False
            if "usecase" not in uml_code:
                logger.error("No use cases defined")
                return False
            if "rectangle" not in uml_code:
                logger.error("No system boundary defined")
                return False
                
            # Try to parse the UML code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.puml', delete=False) as temp_file:
                temp_file.write(uml_code)
                temp_file_path = temp_file.name
                
            try:
                # Test if the code can be processed
                plantuml.PlantUML(url=self.plantuml_server).processes_file(
                    temp_file_path, outfile=os.devnull
                )
                return True
            except Exception as e:
                logger.error(f"PlantUML syntax error: {str(e)}")
                return False
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error validating UML: {str(e)}")
            return False
            
    def generate_diagram(self, uml_code: str, output_path: str):
        """Generate diagram from PlantUML code."""
        try:
            # Create a temporary file with the UML code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.puml', delete=False) as temp_file:
                temp_file.write(uml_code)
                temp_file_path = temp_file.name
            
            try:
                # Generate the diagram
                plantuml.PlantUML(url=self.plantuml_server).processes_file(
                    temp_file_path, outfile=output_path
                )
                logger.info(f"Diagram generated at {output_path}")
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error generating diagram: {str(e)}")
            raise 