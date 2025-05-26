import logging
from typing import Dict, Any, Optional, List
import plantuml
import os
from dotenv import load_dotenv
from neo4j_operations import Neo4jOperations
from llm_client import LLMClient
import re
import subprocess
import json
import traceback
import uuid
import time
import pickle

# Configure logging (basicConfig call removed, logger will inherit from root)
logger = logging.getLogger(__name__)

class PlantUMLGenerator:
    """
    Generate PlantUML diagrams from extracted entities and relationships.
    This class handles the generation of various UML diagrams using PlantUML syntax.
    """
    
    def __init__(self, neo4j_ops):
        """
        Initialize the PlantUML generator.
        
        Args:
            neo4j_ops: Neo4j operations object
        """
        self.neo4j_ops = neo4j_ops
        self.llm_client = LLMClient()
        load_dotenv()
        self.server_url = os.getenv("PLANTUML_SERVER_URL", "http://www.plantuml.com/plantuml/img/")
        self.plantuml_client = plantuml.PlantUML(url=self.server_url)
        
        # Create output directory structure
        self.base_output_dir = os.getenv("OUTPUT_DIR", "output")
        self.diagrams_dir = os.path.join(self.base_output_dir, "diagrams")
        self.plantuml_dir = os.path.join(self.base_output_dir, "plantuml")
        self.neo4j_dir = os.path.join(self.base_output_dir, "neo4j")
        
        # Create directories
        for directory in [self.base_output_dir, self.diagrams_dir, self.plantuml_dir, self.neo4j_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Add request throttling to avoid rate limits
        self.last_request_time = 0
        self.min_request_interval = 3  # Minimum seconds between requests
        
        # Free tier optimization
        self.optimize_for_free_tier = False

    def _throttled_llm_request(self, prompt, max_tokens=2000, temperature=0.1):
        """Make an LLM request with throttling to avoid rate limits."""
        import time
        
        # Calculate time since last request
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # If we need to wait to respect rate limits
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            print(f"Rate limit protection: Waiting {wait_time:.2f}s between requests...")
            time.sleep(wait_time)
            
        # Apply free tier optimizations if enabled
        if self.optimize_for_free_tier:
            # Use template generation instead of LLM when possible
            if "generate plantuml" in prompt.lower() and hasattr(self, '_last_analysis_result'):
                print("Free tier optimization: Using template generation instead of LLM")
                return self._generate_template_plantuml(self._last_analysis_result)
                
        # Make the request
        response = self.llm_client.generate_text(prompt, max_tokens, temperature)
        
        # Update the last request time
        self.last_request_time = time.time()
        
        return response
        
    def _optimize_prompt_for_free_tier(self, prompt):
        """Optimize prompts to be more concise and token-efficient for free tier."""
        # We're no longer optimizing prompts - just return the original
        return prompt

    def generate_plantuml(self, analysis_result: Dict[str, Any]) -> Optional[str]:
        """
        Generate PlantUML code primarily from analysis_result (entities.json)
        with KG used only for validation.
        
        Args:
            analysis_result (Dict[str, Any]): Analysis results from LLM
        
        Returns:
            Optional[str]: PlantUML code or None if generation fails
        """
        try:
            print("Starting PlantUML generation from entities.json with KG validation")
            
            # Store for potential template-based generation in free tier
            self._last_analysis_result = analysis_result
            
            # For free tier, prefer template generation to save API calls
            if self.optimize_for_free_tier:
                print("Free tier optimization: Using template generation for PlantUML")
                uml_code = self._generate_uml_code(analysis_result)
                if uml_code:
                    return uml_code
                # Fall back to LLM if template fails
                print("Template generation failed, falling back to LLM (limited by free tier)")
            
            # Step 1: Get validation patterns from KG
            kg_patterns = self._get_kg_patterns()
            print(f"Retrieved {len(kg_patterns)} validation patterns from KG")
            
            # Save KG patterns to file for debugging
            try:
                patterns_file = os.path.join(self.plantuml_dir, "kg_patterns_debug.txt")
                with open(patterns_file, 'w', encoding='utf-8') as f:
                    f.write(f"KG Patterns Retrieved: {len(kg_patterns)} patterns\n")
                    f.write("=" * 50 + "\n\n")
                    for i, pattern in enumerate(kg_patterns):
                        f.write(f"Pattern {i+1}:\n")
                        f.write(f"Type: {type(pattern)}\n")
                        if isinstance(pattern, dict):
                            f.write(f"Content: {json.dumps(pattern, indent=2)}\n")
                        else:
                            f.write(f"Content: {pattern}\n")
                        f.write("-" * 30 + "\n")
                print(f"Saved KG patterns debug info to: {patterns_file}")
            except Exception as e:
                print(f"Could not save KG patterns debug file: {e}")
            
            # Step 2: Generate PlantUML code directly from entities.json
            uml_code = self._generate_plantuml_with_llm(analysis_result, kg_patterns)
            
            if not uml_code:
                print("Failed to generate UML code with LLM")
                # Fallback to template-based generation
                uml_code = self._generate_uml_code(analysis_result)
            
            if not uml_code:
                print("Failed to generate UML code")
                return None
            
            # Step 3: Use KG to validate the generated code
            if self.neo4j_ops and not self.optimize_for_free_tier:  # Skip KG validation in free tier to save API calls
                validation_result = self._validate_with_kg(analysis_result, uml_code)
                if not validation_result["is_valid"]:
                    print(f"KG validation found issues: {validation_result['issues']}")
                    # We could use these issues to improve the code but will continue anyway
            
            # Step 4: Validate the syntax of the generated code
            validation_result, error_info = self._validate_uml_syntax(uml_code)
            if not validation_result:
                print("Invalid UML code, trying to fix with LLM")
                # Try to fix with LLM, passing the error information
                uml_code = self._fix_plantuml_syntax_with_llm(uml_code, error_info)
                validation_result, error_info = self._validate_uml_syntax(uml_code)
                if not validation_result:
                    # Still invalid, try template generation
                    print("Still invalid after LLM fix. Falling back to template generation.")
                    uml_code = self._generate_uml_code(analysis_result)
                    if not uml_code:
                        return None
            
            return uml_code

        except Exception as e:
            print(f"Error in PlantUML generation: {str(e)}")
            print(traceback.format_exc())
            # Try template-based generation as last resort
            try:
                return self._generate_uml_code(analysis_result)
            except:
                return None

    def _fetch_kg_facts(self) -> Dict[str, Any]:
        """
        Fetch verified facts from the Neo4j knowledge graph with enhanced validation.
        
        Returns:
            Dict[str, Any]: Dictionary containing verified actors, use cases, and relationships
        """
        try:
            facts = {
                "actors": [],
                "use_cases": [],
                "relationships": [],
                "system": None,
                "patterns": [],
                "validation_rules": []
            }
            
            # Get system information
            with self.neo4j_ops.driver.session() as session:
                # Get system info
                system_result = session.run("""
                    MATCH (s:System)
                    RETURN s.name as name, s.description as description
                    LIMIT 1
                """)
                system_data = system_result.data()
                if system_data:
                    facts["system"] = system_data[0]
                
                # Get actors with validation rules
                actor_result = session.run("""
                    MATCH (a:Actor)
                    RETURN a.name as name, a.type as type, a.description as description,
                           a.validation_rules as validation_rules
                """)
                facts["actors"] = actor_result.data()
                
                # Get use cases with patterns
                usecase_result = session.run("""
                    MATCH (u:UseCase)
                    RETURN u.name as name, u.description as description,
                           u.priority as priority, u.preconditions as preconditions,
                           u.postconditions as postconditions,
                           u.patterns as patterns
                """)
                facts["use_cases"] = usecase_result.data()
                
                # Get relationship patterns
                pattern_result = session.run("""
                    MATCH (p:Pattern)
                    RETURN p.name as name, p.description as description,
                           p.validation_rules as validation_rules
                """)
                facts["patterns"] = pattern_result.data()
                
                # Get all relationships with validation
                actor_rels = session.run("""
                    MATCH (a:Actor)-[r]->(u:UseCase)
                    RETURN a.name as from, u.name as to, type(r) as type,
                           r.validation_rules as validation_rules
                """)
                
                usecase_rels = session.run("""
                    MATCH (u1:UseCase)-[r]->(u2:UseCase)
                    RETURN u1.name as from, u2.name as to, type(r) as type,
                           r.validation_rules as validation_rules
                """)
                
                facts["relationships"] = actor_rels.data() + usecase_rels.data()
                
                # Get validation rules
                validation_result = session.run("""
                    MATCH (r:ValidationRule)
                    RETURN r.name as name, r.description as description,
                           r.conditions as conditions
                """)
                facts["validation_rules"] = validation_result.data()
            
            return facts
            
        except Exception as e:
            print(f"Error fetching facts from knowledge graph: {str(e)}")
            return {
                "actors": [], 
                "use_cases": [], 
                "relationships": [], 
                "system": None,
                "patterns": [],
                "validation_rules": []
            }
    
    def _combine_kg_and_analysis(self, kg_facts: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine knowledge graph facts with analysis results, prioritizing KG facts.
        
        Args:
            kg_facts (Dict[str, Any]): Facts from knowledge graph
            analysis_result (Dict[str, Any]): Analysis results from LLM
            
        Returns:
            Dict[str, Any]: Combined facts
        """
        combined = {
            "actors": [],
            "use_cases": [],
            "relationships": [],
            "system": kg_facts.get("system") or analysis_result.get("system", {"name": "System"})
        }
        
        # Process actors - add KG actors first, then add new actors from analysis
        kg_actor_names = set(a.get("name", "").lower() for a in kg_facts.get("actors", []))
        combined["actors"].extend(kg_facts.get("actors", []))
        
        # Add new actors from analysis_result that aren't in KG
        for actor in analysis_result.get("actors", []):
            if actor.get("name", "").lower() not in kg_actor_names:
                combined["actors"].append(actor)
        
        # Process use cases - add KG use cases first, then add new use cases from analysis
        kg_usecase_names = set(uc.get("name", "").lower() for uc in kg_facts.get("use_cases", []))
        combined["use_cases"].extend(kg_facts.get("use_cases", []))
        
        # Add new use cases from analysis_result that aren't in KG
        for usecase in analysis_result.get("use_cases", []):
            if usecase.get("name", "").lower() not in kg_usecase_names:
                combined["use_cases"].append(usecase)
        
        # Process relationships - add KG relationships first, then add new relationships from analysis
        combined["relationships"].extend(kg_facts.get("relationships", []))
        kg_rel_keys = set((r.get("from", "").lower(), r.get("to", "").lower()) for r in kg_facts.get("relationships", []))
        
        # Add new relationships from analysis_result that aren't in KG
        for rel in analysis_result.get("relationships", []):
            if (rel.get("from", "").lower(), rel.get("to", "").lower()) not in kg_rel_keys:
                combined["relationships"].append(rel)
        
        return combined
    
    def _generate_plantuml_with_llm(self, facts: Dict[str, Any], kg_patterns: List) -> Optional[str]:
        """
        Generate PlantUML code using LLM, working directly with entities.json
        and incorporating KG patterns for guidance.
        
        Args:
            facts: Dictionary containing actors, use cases, and relationships
            kg_patterns: List of patterns from knowledge graph for validation
            
        Returns:
            Optional[str]: Generated PlantUML code or None if generation fails
        """
        try:
            system_name = facts.get("system", {}).get("name") or facts.get("system", {}).get("scope") or "System"
            actors = facts.get("actors", [])
            use_cases = facts.get("use_cases", [])
            relationships = facts.get("relationships", [])

            # Get actor and use case names for the prompt
            actor_names = [a.get('name', a) if isinstance(a, dict) else a for a in actors]
            usecase_names = [uc.get('name', uc) if isinstance(uc, dict) else uc for uc in use_cases]
            
            # Format relationships for the prompt
            rel_formatted = []
            for rel in relationships:
                source = rel.get("source", rel.get("from", "Source"))
                target = rel.get("target", rel.get("to", "Target"))
                rel_type = rel.get("type", "association").lower()
                
                if rel_type == "association":
                    rel_formatted.append(f"{source} ---> {target}")
                elif rel_type == "include":
                    rel_formatted.append(f"{source} ..> {target} : <<include>>")
                elif rel_type == "extend":
                    rel_formatted.append(f"{source} ..> {target} : <<extend>>")
                elif rel_type in ["generalization", "inheritance"]:
                    rel_formatted.append(f"{source} --|> {target}")
                else:
                    rel_formatted.append(f"{source} --- {target} : {rel_type}")
            
            # Format KG patterns for guidance
            pattern_guidance = ""
            if kg_patterns:
                pattern_guidance = "Follow these UML design patterns from our knowledge base:\n"
                for i, pattern in enumerate(kg_patterns):
                    if isinstance(pattern, dict):
                        pattern_text = pattern.get("description", pattern.get("name", ""))
                    else:
                        pattern_text = str(pattern)
                    if pattern_text:
                        pattern_guidance += f"  {i+1}. {pattern_text}\n"

            # Compose the prompt with additional guidance
            prompt = f"""
Generate valid PlantUML code for a use case diagram following these exact requirements:

1. The code MUST start with @startuml and end with @enduml
2. Use 'left to right direction' for better readability
3. Declare actors with 'actor "Name"'
4. Define use cases inside a system boundary rectangle with 'rectangle "System" {{ ... }}'
5. Define use cases with 'usecase "UseCase"' or '(UseCase)'
6. For relationships:
   - Association between actors and use cases: actor --> usecase
   - Include relationship: usecase1 ..> usecase2 : <<include>>
   - Extend relationship: usecase1 ..> usecase2 : <<extend>>
   - Generalization (inheritance): actor1 --|> actor2 (for actors)
   - Generalization (inheritance): usecase1 --|> usecase2 (for use cases)
7. Use proper indentation for elements inside the system boundary

{pattern_guidance}

For this specific system, use the following elements:
- System Name: {system_name}
- Actors: {actor_names}
- Use Cases: {usecase_names}
- Relationships: 
{chr(10).join(rel_formatted)}

Example:
```
@startuml
title System Use Case Diagram
left to right direction
skinparam actorStyle awesome
skinparam usecaseStyle oval
skinparam packageStyle rectangle

actor "User"
actor "Admin"

rectangle "System" {{
  usecase "Login"
  usecase "Manage Data"
  usecase "View Reports"
  
  "User" --> "Login"
  "Admin" --> "Login"
  "Admin" --> "Manage Data"
  "Login" ..> "View Reports" : <<include>>
}}


@enduml
```

Please create a use case diagram for {system_name} with the provided actors, use cases, and relationships.
"""
            print("Sending UML generation prompt to LLM...")
            print(f"Prompt contains {len(prompt)} characters")
            llm_response = self._throttled_llm_request(prompt)
            
            if not llm_response:
                print("LLM returned no response for PlantUML generation.")
                return None
                
            print(f"Received LLM response of {len(llm_response)} characters")
            extracted_code = self._extract_pure_plantuml_code(llm_response)
            
            if not extracted_code:
                print("Failed to extract PlantUML code from LLM response.")
                print(f"LLM Raw Response was:\n{llm_response[:200]}...")
                return None
                
            print("Successfully generated PlantUML code with LLM.")
            return extracted_code
            
        except Exception as e:
            print(f"Error generating PlantUML code with LLM: {str(e)}")
            print(traceback.format_exc())
            return None

    def _extract_pure_plantuml_code(self, llm_output: str) -> Optional[str]:
        """Extracts PlantUML code from a string, typically LLM output with markdown."""
        if not llm_output:
            return None

        # Try to find @startuml and @enduml
        match = re.search(r"@startuml(.*?)@enduml", llm_output, re.DOTALL)
        if match:
            return f"@startuml{match.group(1).strip()}\n@enduml"

        # Try to find ```plantuml ... ``` or ``` ... ``` blocks
        # More specific ```plantuml pattern
        match = re.search(r"```plantuml\s*(.*?)\s*```", llm_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Generic ``` code block pattern (less reliable, might pick up other code)
        match = re.search(r"```\s*(.*?)\s*```", llm_output, re.DOTALL)
        if match:
            # Check if the content looks like PlantUML (heuristic)
            content = match.group(1).strip()
            if "actor" in content or "usecase" in content or "-->" in content or "<|--" in content:
                return content
        
        # If no explicit markers, assume the whole output might be PlantUML if it's not too long
        # and contains PlantUML keywords. This is a fallback.
        if len(llm_output.splitlines()) < 50 and ("@startuml" not in llm_output and "@enduml" not in llm_output):
             if "actor" in llm_output or "usecase" in llm_output or "-->" in llm_output:
                 return llm_output.strip()

        print("Could not reliably extract PlantUML code from LLM output.")
        return None # Or return llm_output if we want to try it anyway

    def _validate_plantuml_code(self, code: str, guidance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated PlantUML code against textbook rules.
        
        Args:
            code: Generated PlantUML code
            guidance: Dictionary containing textbook guidance
            
        Returns:
            Dict[str, Any]: Validation result with issues if any
        """
        issues = []
        
        # Check for required elements
        if 'actor' not in code.lower():
            issues.append("Missing actor definitions")
        if 'usecase' not in code.lower():
            issues.append("Missing use case definitions")
        
        # Check relationship syntax
        relationship_patterns = {
            'include': r'\.\.\s*<<\s*include\s*>>',
            'extend': r'\.\.\s*<<\s*extend\s*>>',
            'generalization': r'--\|>',
            'association': r'--'
        }
        
        for rel_type, pattern in relationship_patterns.items():
            if rel_type in guidance['relationships'] and not re.search(pattern, code):
                issues.append(f"Missing or incorrect {rel_type} relationship syntax")
            
        # Check notation rules
        for rule in guidance['notation']:
            if 'actor' in rule.lower() and ':' not in code:
                issues.append("Missing actor notation")
            if 'usecase' in rule.lower() and '()' not in code:
                issues.append("Missing use case notation")
            
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }

    def _fix_plantuml_syntax_with_llm(self, uml_code: str, error_info: str = "") -> str:
        """
        Fix PlantUML syntax errors using LLM.
        
        Args:
            uml_code (str): PlantUML code with syntax errors
            error_info (str): Information about the error that caused validation to fail
            
        Returns:
            str: Fixed PlantUML code
        """
        try:
            print("===== ATTEMPTING TO FIX PLANTUML SYNTAX WITH LLM =====")
            print(f"Original UML code length: {len(uml_code)} characters")
            print(f"Error information: {error_info}")
            
            # Prepare a prompt with the error information
            error_guidance = ""
            if error_info:
                error_guidance = f"The code has the following specific errors that need to be fixed:\n{error_info}\n\n"
            
            prompt = f"""Fix the syntax errors in this PlantUML use case diagram code.

{error_guidance}This diagram MUST follow standard use case diagram notation:

1. CORRECT SYNTAX FOR USE CASE DIAGRAMS:
   - The code must start with @startuml and end with @enduml
   - Use case notation: usecase "Name" or (Name)
   - Actor notation: actor "Name"
   - System boundary: rectangle "System" {{ ... }}
   - Relationships:
     * Association (actor to use case): actor --> usecase
     * Include: usecase1 ..> usecase2 : <<include>>
     * Extend: usecase1 ..> usecase2 : <<extend>>
     * Generalization: actor1 --|> actor2 or usecase1 --|> usecase2

2. COMMON ERRORS TO FIX:
   - Missing @startuml or @enduml tags
   - Incorrect relationship syntax (should use ..> for include/extend with stereotypes)
   - Invalid use of package instead of rectangle for system boundary
   - Missing or incorrect stereotypes (<<include>>, <<extend>>)
   - Invalid actor or usecase names (need quotes for multi-word names)
   - Unbalanced braces or parentheses
   - Incorrect arrow directions

PlantUML code with errors:
```
{uml_code}
```

Return ONLY the corrected PlantUML code with no additional explanations or comments.
Make sure the output is valid PlantUML with proper @startuml and @enduml tags."""
            
            print("Sending prompt to LLM to fix UML code...")
            # Use throttled request instead of direct LLM call
            response = self._throttled_llm_request(prompt, max_tokens=1500, temperature=0.1)
            print(f"LLM response received, length: {len(response)} characters")
            
            # Extract code from response
            if "@startuml" in response and "@enduml" in response:
                start = response.find("@startuml")
                end = response.find("@enduml") + len("@enduml")
                fixed_code = response[start:end]
                print("Successfully fixed PlantUML syntax with LLM")
                print(f"Fixed UML code length: {len(fixed_code)} characters")
                print(f"First 100 chars of fixed UML code: {fixed_code[:100]}...")
                
                # Save the fixed UML code to a debug file
                debug_file_path = os.path.join(self.plantuml_dir, "llm_fixed_uml_code.txt")
                with open(debug_file_path, 'w', encoding='utf-8') as debug_file:
                    debug_file.write(fixed_code)
                print(f"Saved LLM-fixed UML code for debugging to: {debug_file_path}")
                
                # Check for common PlantUML syntax elements in the fixed code
                has_actors = "actor" in fixed_code.lower()
                has_usecases = "usecase" in fixed_code.lower() or "(" in fixed_code
                has_relationships = "-->" in fixed_code or ".." in fixed_code
                print(f"Fixed code contains: actors: {has_actors}, usecases: {has_usecases}, relationships: {has_relationships}")
                
                return fixed_code
            else:
                print("WARNING: LLM response doesn't contain valid PlantUML code with @startuml/@enduml tags")
                print(f"LLM response preview: {response[:200]}...")
                return uml_code  # Return original code if fixing failed
                
        except Exception as e:
            print(f"ERROR fixing PlantUML syntax with LLM: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            return uml_code  # Return original code if exception occurred

    def _get_kg_patterns(self):
        """
        Get patterns and common structures from knowledge graph.
        
        Returns:
            list: List of patterns
        """
        # Check if Neo4j is connected
        if self.neo4j_ops is None:
            # Return default patterns
            return [
                "Actor should be placed at the left or right of the diagram",
                "Use cases should be inside the system boundary",
                "Include relationships use <<include>> stereotype",
                "Extend relationships use <<extend>> stereotype",
                "Inheritance relationships use generalization arrow",
                "Actor inheritance uses generalization arrow"
            ]
        
        try:
            # Get patterns from knowledge graph or use defaults
            patterns = self.neo4j_ops.get_patterns() or []
            
            # Add default patterns if not enough from KG
            default_patterns = [
                "Actor should be placed at the left or right of the diagram",
                "Use cases should be inside the system boundary",
                "Include relationships use <<include>> stereotype",
                "Extend relationships use <<extend>> stereotype",
                "Inheritance relationships use generalization arrow",
                "Actor inheritance uses generalization arrow"
            ]
            
            if len(patterns) < 3:
                patterns.extend(default_patterns)
            
            return patterns
            
        except Exception as e:
            print(f"Error getting patterns: {str(e)}")
            # Return default patterns on error
            return [
                "Actor should be placed at the left or right of the diagram",
                "Use cases should be inside the system boundary",
                "Include relationships use <<include>> stereotype",
                "Extend relationships use <<extend>> stereotype",
                "Inheritance relationships use generalization arrow",
                "Actor inheritance uses generalization arrow"
            ]

    def _generate_uml_code(self, validated_entities: Dict[str, Any]) -> Optional[str]:
        """
        Generate PlantUML code from validated entities (template-based as a fallback).
        Args:
            validated_entities: Dictionary of actors, use cases, and relationships.
        Returns:
            PlantUML code as a string or None if generation fails.
        """
        try:
            actors = validated_entities.get("actors", [])
            use_cases = validated_entities.get("use_cases", [])
            relationships = validated_entities.get("relationships", [])
            system_name = validated_entities.get("system", {}).get("name", "System")

            plantuml_code = ["@startuml"]
            plantuml_code.append(f"title Use Case Diagram for {system_name}\n")
            plantuml_code.append("left to right direction\n") # Default direction
            # Define skinparams for better appearance
            plantuml_code.append("skinparam packageStyle rectangle")
            plantuml_code.append("skinparam actorStyle awesome") # Modern actor style
            plantuml_code.extend([
                "skinparam usecase {",
                "    BackgroundColor PaleGreen",
                "    BorderColor Green",
                "    ArrowColor Green",
                "}"
            ])
            plantuml_code.extend([
                "skinparam rectangle {",
                "    BackgroundColor LightBlue",
                "    BorderColor Blue",
                "}"
            ])

            # Define actors
            if actors:
                for actor_item in actors:
                    actor_name = self._sanitize_name(actor_item.get("name"))
                    if actor_name:
                        plantuml_code.append(f"actor \"{actor_name}\"")
            else:
                plantuml_code.append("actor \"PlaceholderActor\" as PA") # Placeholder if no actors

            # Define use cases within a system boundary
            plantuml_code.append(f"rectangle \"{self._sanitize_name(system_name)}\" {{")
            if use_cases:
                for uc_item in use_cases:
                    uc_name = self._sanitize_name(uc_item.get("name"))
                    if uc_name:
                        plantuml_code.append(f"  usecase \"{uc_name}\"")
            else:
                plantuml_code.append("  usecase \"PlaceholderUseCase\" as PUC") # Placeholder
            plantuml_code.append("}")

            # Define relationships
            if relationships:
                for rel in relationships:
                    # These keys 'from' and 'to' come from _fetch_kg_facts or llm_extractor
                    from_node_name = self._sanitize_name(rel.get("from")) 
                    to_node_name = self._sanitize_name(rel.get("to"))
                    rel_type = rel.get("type", "").lower()
                    description = rel.get("description", "")

                    if not from_node_name or not to_node_name:
                        print(f"Skipping relationship due to missing from/to node: {rel}")
                        continue

                    # Determine if 'from_node' is an actor or use case
                    # This is a heuristic; ideally, the type of 'from_node' and 'to_node' should be known.
                    from_is_actor = any(actor.get("name") == rel.get("from") for actor in actors)
                    to_is_actor = any(actor.get("name") == rel.get("to") for actor in actors) # Though use cases usually don't point to actors

                    from_node_q = f'\"{from_node_name}\"' if not from_is_actor else f'\"{from_node_name}\"' # Actors are already quoted if needed by actor definition
                    to_node_q = f'\"{to_node_name}\"' 
                    
                    # Actors are typically outside the rectangle, use cases inside.
                    # Standard actor-usecase relationship:
                    if from_is_actor and not to_is_actor:
                        plantuml_code.append(f"  {from_node_q} --> {to_node_q}{': ' + description if description else ''}")
                    # Use case to Use case relationships (include, extend)
                    elif not from_is_actor and not to_is_actor: 
                        if "include" in rel_type:
                            plantuml_code.append(f"  {from_node_q} ..> {to_node_q} : <<include>>")
                        elif "extend" in rel_type:
                            plantuml_code.append(f"  {from_node_q} <.. {to_node_q} : <<extend>>")
                        else: # General association between use cases
                            plantuml_code.append(f"  {from_node_q} -- {to_node_q}{': ' + description if description else ''}")
                    else:
                        # Other relationships or fallback
                        plantuml_code.append(f"  {from_node_q} -- {to_node_q}{': ' + rel_type + ' ' + description if description else (': ' + rel_type if rel_type else '')}")
            
            plantuml_code.append("@enduml")
            return "\n".join(plantuml_code)
        except Exception as e:
            print(f"Error generating template UML code: {str(e)}")
            print(traceback.format_exc())
            return None

    def _validate_uml_syntax(self, uml_code: str) -> tuple:
        """
        Validate PlantUML syntax.
        
        Args:
            uml_code (str): PlantUML code to validate
            
        Returns:
            tuple: (bool, str) - (True if syntax is valid, error information if any)
        """
        error_info = ""
        try:
            if not uml_code:
                print("Empty UML code")
                return False, "Empty UML code"
            
            # Save the full UML code to a debug file before validation
            debug_file_path = os.path.join(self.plantuml_dir, "pre_validation_uml_code.txt")
            with open(debug_file_path, 'w', encoding='utf-8') as debug_file:
                debug_file.write(uml_code)
            print(f"Saved UML code for debugging to: {debug_file_path}")
            
            # Check for basic syntax elements
            if not uml_code.startswith("@startuml") or not uml_code.endswith("@enduml"):
                error_info = "Missing @startuml or @enduml tags"
                print(error_info)
                return False, error_info
                
            # Check for balanced brackets
            open_brackets = uml_code.count("{")
            close_brackets = uml_code.count("}")
            if open_brackets != close_brackets:
                error_info = f"Unbalanced brackets: {open_brackets} opening vs {close_brackets} closing"
                print(error_info)
                return False, error_info
                
            # Check for common syntax errors
            error_patterns = [
                "actor acting", "actor actor", "usecase usecase",
                "---->", "<---->", "----", "--->", "<----",
                "<<extends>>", "<<generalization>>", "<<inherits>>"
            ]
            
            for pattern in error_patterns:
                if pattern in uml_code:
                    error_info = f"Found error pattern: {pattern}"
                    print(error_info)
                    return False, error_info
                    
            # Try to generate a diagram as the final validation
            try:
                temp_file = os.path.join(self.plantuml_dir, "syntax_check.puml")
                # Ensure we're writing string data with proper encoding
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(uml_code)
                
                print(f"Created temporary validation file at {temp_file}")
                print(f"First 100 chars of UML code: {uml_code[:100]}...")
                
                # This will throw an exception if there are syntax errors
                try:
                    print("Attempting to process UML code with PlantUML client...")
                    # Process the UML code and get binary content
                    content = self.plantuml_client.processes(uml_code)
                    
                    print(f"PlantUML response received, type: {type(content)}, size: {len(content) if content else 'empty'}")
                    
                    # For validation purposes, we can just check if content is valid
                    # without writing it to a file, but if we need to save it:
                    validation_output_path = os.path.join(self.plantuml_dir, "validation_check.png")
                    with open(validation_output_path, 'wb') as f:  # Note: 'wb' for binary write
                        f.write(content)
                    
                    print(f"Saved validation output to {validation_output_path}")
                    
                    # If we got here, the syntax is valid
                    # Remove temp file if validation successful
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
                    print("PlantUML syntax validation passed")
                    return True, ""
                except Exception as e:
                    # Handle any error from PlantUML client
                    error_info = f"PlantUML validation error: {str(e)}"
                    print(f"DETAILED ERROR: {error_info}")
                    print(f"Exception type: {type(e).__name__}")
                    print(f"Traceback: {traceback.format_exc()}")
                    
                    # Try to capture relevant error details
                    error_details_path = os.path.join(self.plantuml_dir, "validation_error_details.txt")
                    with open(error_details_path, 'w', encoding='utf-8') as f:
                        f.write(f"Error type: {type(e).__name__}\n")
                        f.write(f"Error message: {str(e)}\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                        f.write("\n--- UML Code that caused error ---\n")
                        f.write(uml_code)
                    print(f"Saved validation error details to: {error_details_path}")
                    
                    return False, error_info
                
            except plantuml.PlantUMLHTTPError as e:
                # Handle the specific PlantUMLHTTPError without assuming attributes
                try:
                    # Safely access response attributes
                    status = getattr(e.response, 'status', 'unknown')
                    reason = getattr(e.response, 'reason', 'unknown')
                    error_info = f"PlantUML HTTP Error: Status {status} - {reason}"
                    print(f"DETAILED HTTP ERROR: {error_info}")
                except AttributeError:
                    # Fallback if response or its attributes aren't available
                    error_info = f"PlantUML HTTP Error: {str(e)}"
                    print(f"DETAILED HTTP ERROR (no details): {str(e)}")
                
                try:
                    # Save HTTP error details to a readable text file
                    http_error_path = os.path.join(self.plantuml_dir, "http_error_details.txt")
                    with open(http_error_path, 'w', encoding='utf-8') as f:
                        f.write(f"HTTP Error: {str(e)}\n")
                        try:
                            f.write(f"Status: {getattr(e.response, 'status', 'unknown')}\n")
                            f.write(f"Reason: {getattr(e.response, 'reason', 'unknown')}\n")
                        except AttributeError:
                            f.write("Could not access response attributes\n")
                        
                        f.write("\n--- UML Code that caused error ---\n")
                        f.write(uml_code)
                    print(f"Saved HTTP error details to: {http_error_path}")
                    
                    # Safely get and save error content
                    error_content = getattr(e, 'content', None)
                    if error_content:
                        error_content_path = os.path.join(self.plantuml_dir, "error_content.html")
                        with open(error_content_path, 'wb') as f:  # Note: 'wb' for binary write
                            f.write(error_content)
                        print(f"Error content saved to {error_content_path}")
                        
                        # Try to extract useful info from the error content for the LLM
                        try:
                            error_text = error_content.decode('utf-8')
                            print(f"Error content preview: {error_text[:200]}...")
                            if "syntax error" in error_text.lower():
                                print("Syntax error detected in PlantUML code")
                                # Extract more specific error details if possible
                                error_lines = [line for line in error_text.split('\n') if 'error' in line.lower() or 'warning' in line.lower()]
                                if error_lines:
                                    print(f"Specific error messages: {error_lines}")
                            elif "cannot find" in error_text.lower():
                                print("Missing element reference detected in PlantUML code")
                        except Exception as decode_err:
                            print(f"Could not decode error content: {str(decode_err)}")
                    else:
                        print("No error content available in the PlantUML HTTP error")
                except Exception as content_err:
                    print(f"Error accessing HTTP error content: {str(content_err)}")
                
                return False, error_info
                
            except Exception as e:
                error_info = f"PlantUML syntax validation failed: {str(e)}"
                print(f"VALIDATION EXCEPTION: {error_info}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Traceback: {traceback.format_exc()}")
                return False, error_info
                
        except Exception as e:
            error_info = f"Error validating UML syntax: {str(e)}"
            print(f"OUTER EXCEPTION: {error_info}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            return False, error_info

    def generate_diagram(self, uml_code: str, output_path: str) -> bool:
        """
        Generate diagram image from PlantUML code.
        
        Args:
            uml_code (str): PlantUML code
            output_path (str): Path to save generated diagram
            
        Returns:
            bool: True if generation is successful, False otherwise
        """
        try:
            print("===== GENERATE DIAGRAM =====")
            print(f"Generating diagram at: {output_path}")
            print(f"UML code length: {len(uml_code)} characters")
            print(f"UML code preview: {uml_code[:100]}...")
            
            # Create a temporary file for the code
            temp_file = os.path.join(self.plantuml_dir, "temp.puml")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(uml_code)
                
            print(f"Wrote UML code to temporary file: {temp_file}")
                
            try:
                # Generate the diagram
                print("Calling PlantUML client to process UML code...")
                content = self.plantuml_client.processes(uml_code)
                
                # Check if content is bytes (which it should be)
                if not isinstance(content, bytes):
                    print(f"WARNING: PlantUML processes returned non-bytes type: {type(content)}")
                    # Convert to bytes if possible
                    if isinstance(content, str):
                        print("Converting string content to bytes...")
                        content = content.encode('utf-8')
                else:
                    print(f"Received binary content from PlantUML, size: {len(content)} bytes")
                
                # Write binary content directly to output file
                print(f"Writing diagram to: {output_path}")
                with open(output_path, 'wb') as f:
                    f.write(content)
                
                print(f"Successfully generated diagram at {output_path}")
                if os.path.exists(output_path):
                    print(f"Diagram file size: {os.path.getsize(output_path)} bytes")
                return True
                
            except plantuml.PlantUMLHTTPError as e:
                # Handle the specific PlantUMLHTTPError without assuming attributes
                try:
                    # Safely access response attributes
                    status = getattr(e.response, 'status', 'unknown')
                    reason = getattr(e.response, 'reason', 'unknown')
                    print(f"PLANTUML HTTP ERROR: Status {status} - {reason}")
                except AttributeError:
                    # Fallback if response or its attributes aren't available
                    print(f"PLANTUML HTTP ERROR (no details): {str(e)}")
                
                try:
                    # Safely get and save error content
                    error_content = getattr(e, 'content', None)
                    if error_content:
                        error_content_path = os.path.join(self.plantuml_dir, "diagram_error.html")
                        with open(error_content_path, 'wb') as f:
                            f.write(error_content)
                        print(f"Diagram error content saved to {error_content_path}")
                        
                        # Try to extract more detailed error information
                        try:
                            error_text = error_content.decode('utf-8')
                            print(f"Error content preview: {error_text[:200]}...")
                            
                            # Extract specific error lines
                            error_lines = [line for line in error_text.split('\n') if 'error' in line.lower() or 'warning' in line.lower()]
                            if error_lines:
                                print(f"Specific error messages: {error_lines}")
                        except Exception as decode_err:
                            print(f"Could not decode error content: {str(decode_err)}")
                    else:
                        print("No error content available in the PlantUML HTTP error")
                except Exception as content_err:
                    print(f"Error accessing HTTP error content: {str(content_err)}")
                    
                return False
                
            except Exception as e:
                print(f"GENERAL ERROR generating diagram: {str(e)}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Traceback: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            print(f"OUTER ERROR in generate_diagram: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def _format_name(self, name: str) -> str:
        """Format name for PlantUML by removing spaces and special characters."""
        if not name:
            return "unnamed"
        return "".join(c if c.isalnum() else "_" for c in name)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name to prevent PlantUML syntax errors."""
        if not name:
            return "Unnamed"
        return name.replace('"', "'").replace("\\", "/")
    
    def _get_kg_facts(self):
        """
        Get facts from knowledge graph.
        
        Returns:
            dict: Dictionary containing actors and use cases from knowledge graph
        """
        # Check if Neo4j is connected
        if self.neo4j_ops is None:
            print("Neo4j not connected, returning empty facts")
            return {
                "actors": [],
                "use_cases": [],
                "relationships": [],
                "include_relationships": [],
                "extend_relationships": [],
                "generalization_relationships": []
            }
        
        try:
            # Extract actors
            actors = self.neo4j_ops.driver.session().run("""
                MATCH (a:Actor)
                RETURN a.name as name, a.type as type, a.description as description
            """).data()
            
            # Extract use cases
            use_cases = self.neo4j_ops.driver.session().run("""
                MATCH (u:UseCase)
                RETURN u.name as name, u.description as description,
                       u.priority as priority, u.preconditions as preconditions,
                       u.postconditions as postconditions
            """).data()
            
            # Extract relationships
            relationships = self.neo4j_ops.driver.session().run("""
                MATCH (a)-[r]->(u)
                RETURN a.name as from, u.name as to, type(r) as type
            """).data()
            
            # Extract include/extend relationships specifically
            include_relationships = self.neo4j_ops.driver.session().run("""
                MATCH (a)-[r]->(u)
                WHERE type(r) IN ['include', 'INCLUDES']
                RETURN a.name as source, u.name as target
            """).data()
            
            extend_relationships = self.neo4j_ops.driver.session().run("""
                MATCH (a)-[r]->(u)
                WHERE type(r) IN ['extend', 'EXTENDS']
                RETURN a.name as source, u.name as target
            """).data()
            
            generalization_relationships = self.neo4j_ops.driver.session().run("""
                MATCH (a)-[r]->(u)
                WHERE type(r) IN ['generalization', 'GENERALIZATION']
                RETURN a.name as source, u.name as target
            """).data()
            
            print(f"Retrieved facts from knowledge graph: {len(actors)} actors, {len(use_cases)} use cases, " 
                       f"{len(relationships)} relationships, {len(include_relationships)} include, "
                       f"{len(extend_relationships)} extend, {len(generalization_relationships)} generalizations")
            
            return {
                "actors": actors,
                "use_cases": use_cases,
                "relationships": relationships,
                "include_relationships": include_relationships,
                "extend_relationships": extend_relationships,
                "generalization_relationships": generalization_relationships
            }
            
        except Exception as e:
            print(f"Error fetching facts from knowledge graph: {str(e)}")
            return {
                "actors": [],
                "use_cases": [],
                "relationships": [],
                "include_relationships": [],
                "extend_relationships": [],
                "generalization_relationships": []
            }

    def _combine_entities(self, entities, kg_facts):
        """
        Combine entities from initial extraction and knowledge graph.
        
        Args:
            entities (dict): Dictionary containing initially extracted entities
            kg_facts (dict): Dictionary containing entities from knowledge graph
            
        Returns:
            dict: Combined entities
        """
        combined = {
            "actors": [],
            "use_cases": [],
            "relationships": [],
            "include_relationships": [],
            "extend_relationships": [],
            "generalization_relationships": []
        }
        
        # Add actors from both sources (prioritize KG facts)
        actor_names = set()
        for actor in kg_facts.get("actors", []):
            combined["actors"].append(actor)
            actor_names.add(actor.get("name", "").lower())
            
        for actor in entities.get("actors", []):
            if actor.get("name", "").lower() not in actor_names:
                combined["actors"].append(actor)
                actor_names.add(actor.get("name", "").lower())
        
        # Add use cases from both sources (prioritize KG facts)
        use_case_names = set()
        for uc in kg_facts.get("use_cases", []):
            combined["use_cases"].append(uc)
            use_case_names.add(uc.get("name", "").lower())
            
        for uc in entities.get("use_cases", []):
            if uc.get("name", "").lower() not in use_case_names:
                combined["use_cases"].append(uc)
                use_case_names.add(uc.get("name", "").lower())
        
        # Add relationships from both sources
        combined["relationships"] = kg_facts.get("relationships", []) + entities.get("relationships", [])
        
        # Add specific relationships
        combined["include_relationships"] = (
            kg_facts.get("include_relationships", []) + 
            [r for r in entities.get("relationships", []) if r.get("type", "").lower() == "include"]
        )
        
        combined["extend_relationships"] = (
            kg_facts.get("extend_relationships", []) + 
            [r for r in entities.get("relationships", []) if r.get("type", "").lower() == "extend"]
        )
        
        combined["generalization_relationships"] = (
            kg_facts.get("generalization_relationships", []) + 
            [r for r in entities.get("relationships", []) if r.get("type", "").lower() in ["generalization", "inheritance"]]
        )
        
        print(f"Combined entities: {len(combined['actors'])} actors, {len(combined['use_cases'])} use cases, "
                   f"{len(combined['include_relationships'])} include, {len(combined['extend_relationships'])} extend, "
                   f"{len(combined['generalization_relationships'])} generalizations")
                   
        return combined

    def _format_special_relationships_for_prompt(self, relationships, rel_type):
        """Format special relationship types for the LLM prompt."""
        if not relationships:
            return f"No {rel_type} relationships extracted."
        
        formatted = []
        for i, rel in enumerate(relationships):
            source = rel.get("source", "Unknown")
            target = rel.get("target", "Unknown")
            
            formatted.append(f"{i+1}. {source} - {rel_type} -> {target}")
        
        return "\n".join(formatted)

    def _generate_template_plantuml(self, entities):
        """
        Generate PlantUML code using a template approach (fallback).
        
        Args:
            entities (dict): Dictionary containing combined entities
            
        Returns:
            str: PlantUML code
        """
        # Start UML code
        plantuml_code = [
            "@startuml",
            "left to right direction",
            "skinparam actorStyle awesome",
            "skinparam usecaseStyle oval",
            "skinparam packageStyle rectangle",
            "skinparam shadowing false",
            ""
        ]
        
        # Add title
        plantuml_code.append('title Use Case Diagram\n')
        
        # Add actors
        for actor in entities.get("actors", []):
            name = actor.get("name", "").replace('"', '\\"')
            plantuml_code.append(f'actor "{name}"')
        
        plantuml_code.append('')
        
        # Add system boundary
        plantuml_code.append('rectangle "System" {')
        
        # Add use cases
        for uc in entities.get("use_cases", []):
            name = uc.get("name", "").replace('"', '\\"')
            plantuml_code.append(f'  usecase "{name}"')
        
        plantuml_code.append('')
        
        # Process relationships - handle both source/target and from/to key naming conventions
        for rel in entities.get("relationships", []):
            # Get source/from and target/to with fallbacks between different naming conventions
            source = rel.get("source", rel.get("from", "")).replace('"', '\\"')
            target = rel.get("target", rel.get("to", "")).replace('"', '\\"')
            rel_type = rel.get("type", "").lower()
            
            if not source or not target:
                continue  # Skip invalid relationships
                
            # Use proper use case diagram notation
            if rel_type == "include":
                plantuml_code.append(f'  "{source}" ..> "{target}" : <<include>>')
            elif rel_type == "extend":
                plantuml_code.append(f'  "{source}" ..> "{target}" : <<extend>>')
            elif rel_type in ["generalization", "inheritance"]:
                # Use proper generalization for use case diagrams (not component diagrams)
                plantuml_code.append(f'  "{source}" --|> "{target}"')
            else:
                # Standard association
                plantuml_code.append(f'  "{source}" --> "{target}"')
        
        # Add include relationships explicitly
        for rel in entities.get("include_relationships", []):
            source = rel.get("source", rel.get("from", "")).replace('"', '\\"')
            target = rel.get("target", rel.get("to", "")).replace('"', '\\"')
            
            if source and target:
                plantuml_code.append(f'  "{source}" ..> "{target}" : <<include>>')
            
        # Add extend relationships explicitly
        for rel in entities.get("extend_relationships", []):
            source = rel.get("source", rel.get("from", "")).replace('"', '\\"')
            target = rel.get("target", rel.get("to", "")).replace('"', '\\"')
            
            if source and target:
                plantuml_code.append(f'  "{source}" ..> "{target}" : <<extend>>')
            
        # Add generalization relationships explicitly
        for rel in entities.get("generalization_relationships", []):
            source = rel.get("source", rel.get("from", "")).replace('"', '\\"')
            target = rel.get("target", rel.get("to", "")).replace('"', '\\"')
            
            if source and target:
                # Use proper generalization for use case diagrams
                plantuml_code.append(f'  "{source}" --|> "{target}"')
        
        plantuml_code.append("}")
        plantuml_code.append("@enduml")
        return "\n".join(plantuml_code)

    def generate_use_case_diagram(self, entities, output_file):
        """
        Generate a use case diagram using PlantUML syntax and save it.
        
        Args:
            entities (dict): Dictionary containing extracted actors, use cases, and relationships
            output_file (str): Path to save the PlantUML file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate PlantUML code with knowledge graph validation
            plantuml_code = self.generate_plantuml_with_kg_validation(entities)
            
            # Write PlantUML code to file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(plantuml_code)
            
            return True
        
        except Exception as e:
            print(f"Error generating use case diagram: {str(e)}")
            return False 

    def generate_plantuml_with_kg_validation(self, entities):
        """
        Generate PlantUML code with knowledge graph validation.
        
        Args:
            entities (dict): Dictionary containing extracted actors, use cases, and relationships
            
        Returns:
            str: PlantUML code
        """
        print("Starting PlantUML generation with KG validation")
        
        try:
            # Check if Neo4j is connected
            if self.neo4j_ops is None:
                print("Neo4j not connected, using template-based generation")
                return self._generate_template_plantuml(entities)
            
            # Step 1: Get facts from knowledge graph
            kg_facts = self._get_kg_facts()
            
            # Step 2: Get patterns and common structures from knowledge graph
            kg_patterns = self._get_kg_patterns()
            print(f"Retrieved {len(kg_patterns)} patterns from knowledge graph")
            
            # Step 3: Combine entities from initial extraction and knowledge graph
            combined_entities = self._combine_entities(entities, kg_facts)
            
            # Step 4: Use LLM to generate PlantUML code using all the sources of knowledge
            plantuml_code = self._generate_plantuml_with_llm(combined_entities, kg_patterns)
            
            # Step 5: Validate and fix the PlantUML syntax if needed
            is_valid, error_info = self._validate_uml_syntax(plantuml_code)
            
            if not is_valid:
                print(f"PlantUML validation failed: {error_info}")
                # Try to fix with LLM
                plantuml_code = self._fix_plantuml_syntax_with_llm(plantuml_code, error_info)
                
                # Validate again after fix
                is_valid, error_info = self._validate_uml_syntax(plantuml_code)
                if not is_valid:
                    print(f"Still invalid after LLM fix. Falling back to template generation.")
                    plantuml_code = self._generate_template_plantuml(entities)
            
            return plantuml_code
            
        except Exception as e:
            print(f"Error in PlantUML generation with KG validation: {str(e)}")
            # Fallback to template-based generation if KG validation fails
            return self._generate_template_plantuml(entities)

    def generate_diagram_from_plantuml(self, plantuml_file_path: str, output_dir: str) -> bool:
        """
        Generate diagram from an existing PlantUML file.
        
        Args:
            plantuml_file_path (str): Path to the PlantUML file
            output_dir (str): Directory to save the generated diagram
            
        Returns:
            bool: True if generation is successful, False otherwise
        """
        try:
            print("===== GENERATE DIAGRAM =====")
            print(f"Using PlantUML file: {plantuml_file_path}")
            
            # Read the PlantUML file
            with open(plantuml_file_path, 'r', encoding='utf-8') as f:
                uml_code = f.read()
                
            print(f"UML code length: {len(uml_code)} characters")
            print(f"UML code preview: {uml_code[:100]}...")
                
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate the output path
            file_name = os.path.basename(plantuml_file_path)
            output_name = os.path.splitext(file_name)[0] + '.png'
            output_path = os.path.join(output_dir, output_name)
            
            # Validate the UML code syntax before generating
            validation_result, error_info = self._validate_uml_syntax(uml_code)
            if not validation_result:
                print(f"Warning: PlantUML syntax validation failed: {error_info}")
                print("Attempting to fix syntax with LLM...")
                uml_code = self._fix_plantuml_syntax_with_llm(uml_code, error_info)
                
                # Validate again after fix
                validation_result, error_info = self._validate_uml_syntax(uml_code)
                if not validation_result:
                    print(f"Warning: PlantUML syntax still invalid after LLM fix: {error_info}")
                    print("Continuing with diagram generation anyway...")
            
            # Generate the diagram
            success = self.generate_diagram(uml_code, output_path)
            
            if success:
                print(f"Successfully generated diagram at {output_path}")
                print(f"Diagram file size: {os.path.getsize(output_path)} bytes")
            else:
                print(f"Failed to generate diagram")
            
            return success
                
        except Exception as e:
            print(f"Error generating diagram from PlantUML file: {str(e)}")
            print(traceback.format_exc())
            return False

    def _fix_plantuml_code(self, code: str, issues: List[str], guidance: Dict[str, Any]) -> str:
        """
        Attempt to fix validation issues in PlantUML code.
        
        Args:
            code: Original PlantUML code
            issues: List of validation issues
            guidance: Dictionary containing textbook guidance
            
        Returns:
            str: Fixed PlantUML code
        """
        fixed_code = code
        
        for issue in issues:
            if "Missing actor notation" in issue:
                # Add actor notation if missing
                fixed_code = fixed_code.replace('actor', 'actor :')
            elif "Missing use case notation" in issue:
                # Add use case notation if missing
                fixed_code = fixed_code.replace('usecase', 'usecase ()')
            elif "Missing or incorrect" in issue:
                # Fix relationship syntax
                rel_type = issue.split(" ")[-2]
                if rel_type == 'include':
                    fixed_code = fixed_code.replace('..', '.. <<include>>')
                elif rel_type == 'extend':
                    fixed_code = fixed_code.replace('..', '.. <<extend>>')
                elif rel_type == 'generalization':
                    fixed_code = fixed_code.replace('--', '--|>')
            
        return fixed_code

    def _validate_with_kg(self, entities: Dict[str, Any], uml_code: str) -> Dict[str, Any]:
        """
        Validate entities and UML code against knowledge graph patterns.
        Uses KG as:
        1. A Pattern Repository - Validate against UML patterns
        2. A Consistency Checker - Check preconditions/postconditions
        3. A Historical Knowledge Base - Check against previously verified entities
        
        Args:
            entities (Dict[str, Any]): Entities from analysis_result
            uml_code (str): Generated UML code
            
        Returns:
            Dict[str, Any]: Validation result with issues if any
        """
        result = {
            "is_valid": True,
            "issues": []
        }
        
        if not self.neo4j_ops:
            print("Neo4j not connected, skipping KG validation")
            return result
            
        try:
            # 1. Validate against UML patterns in KG
            kg_patterns = self._get_kg_patterns()
            for pattern in kg_patterns:
                if isinstance(pattern, dict):
                    pattern_name = pattern.get("name", "")
                    if pattern_name and pattern_name.lower() not in uml_code.lower():
                        result["issues"].append(f"UML code doesn't follow pattern: {pattern_name}")
                elif isinstance(pattern, str):
                    # Handle string patterns directly
                    if pattern.lower() not in uml_code.lower():
                        result["issues"].append(f"UML code doesn't follow pattern: {pattern}")
            
            # 2. Validate entity consistency against KG
            with self.neo4j_ops.driver.session() as session:
                # Check actors against KG
                for actor in entities.get("actors", []):
                    actor_name = actor.get("name", "")
                    if not actor_name:
                        continue
                        
                    # Check if actor exists in KG
                    query = """
                    MATCH (a:Actor {name: $name})
                    RETURN a.type as type, a.validation_rules as validation_rules
                    """
                    actor_result = session.run(query, name=actor_name).data()
                    
                    if actor_result:
                        # Actor exists, validate consistency
                        kg_type = actor_result[0].get("type", "")
                        actor_type = actor.get("type", "")
                        
                        if kg_type and actor_type and kg_type.lower() != actor_type.lower():
                            result["issues"].append(f"Actor '{actor_name}' type inconsistency: KG={kg_type}, Current={actor_type}")
                    
                # Check use cases against KG
                for use_case in entities.get("use_cases", []):
                    uc_name = use_case.get("name", "")
                    if not uc_name:
                        continue
                        
                    # Check if use case exists in KG
                    query = """
                    MATCH (u:UseCase {name: $name})
                    RETURN u.priority as priority, u.preconditions as preconditions
                    """
                    uc_result = session.run(query, name=uc_name).data()
                    
                    if uc_result:
                        # Use case exists, validate consistency
                        kg_priority = uc_result[0].get("priority", "")
                        uc_priority = use_case.get("priority", "")
                        
                        if kg_priority and uc_priority and kg_priority.lower() != uc_priority.lower():
                            result["issues"].append(f"Use case '{uc_name}' priority inconsistency: KG={kg_priority}, Current={uc_priority}")
                
                # Check relationships against KG
                for rel in entities.get("relationships", []):
                    source = rel.get("source", rel.get("from", ""))
                    target = rel.get("target", rel.get("to", ""))
                    rel_type = rel.get("type", "")
                    
                    if not source or not target or not rel_type:
                        continue
                    
                    # Check if relationship exists in KG
                    query = """
                    MATCH (s)-[r]->(t)
                    WHERE s.name = $source AND t.name = $target
                    RETURN type(r) as rel_type
                    """
                    rel_result = session.run(query, source=source, target=target).data()
                    
                    if rel_result:
                        # Relationship exists, validate type consistency
                        kg_rel_type = rel_result[0].get("rel_type", "")
                        
                        if kg_rel_type and kg_rel_type.lower() != rel_type.lower():
                            result["issues"].append(f"Relationship type inconsistency: KG={kg_rel_type}, Current={rel_type}")
            
            # 3. Set validation status based on issues
            if len(result["issues"]) > 0:
                result["is_valid"] = False
                
            return result
            
        except Exception as e:
            print(f"Error in KG validation: {str(e)}")
            print(traceback.format_exc())
            result["issues"].append(f"KG validation error: {str(e)}")
            result["is_valid"] = False
            return result 