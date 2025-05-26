import json
# import logging # Replaced with print
import re
from typing import Dict, Any, List, Optional
from llm_client import LLMClient
from jsonschema import validate, ValidationError # Added jsonschema

# logger = logging.getLogger(__name__) # Replaced with print

class LLMExtractor:
    def __init__(self):
        self.llm_client = LLMClient()
        self.max_retries = 3
        
        # Add request throttling to avoid rate limits
        self.last_request_time = 0
        self.min_request_interval = 3  # Minimum seconds between requests
        
        # Free tier optimization flags
        self.batch_processing = False  # When True, batch multiple entities in single requests
        self.optimize_prompts = False  # When True, use more token-efficient prompts
        
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
            
        # Make the request
        response = self.llm_client.generate_text(prompt, max_tokens, temperature)
        
        # Update the last request time
        self.last_request_time = time.time()
        
        return response

    def _optimize_prompt_for_free_tier(self, prompt):
        """Optimize prompts to be more concise and token-efficient for free tier."""
        # We're no longer optimizing prompts - just return the original
        return prompt

    def _fix_fr_ids(self, srs: Dict) -> Dict:
        """Pad functional requirement IDs to 3 digits (FR-001, etc) for schema compliance."""
        frs = srs.get("functional_requirements", [])
        for fr in frs:
            if "id" in fr and re.match(r"^FR-\d+$", fr["id"]):
                num = int(fr["id"].split("-")[1])
                fr["id"] = f"FR-{num:03d}"
        return srs

    def generate_standardized_srs(self, text: str) -> Dict:
        """
        Generate a comprehensive SRS document by processing it in logical sections.
        This improves accuracy by allowing the LLM to focus on specific sections at a time.
        
        Steps:
        1. Split input into logical sections with emphasis on UML-critical information
        2. Process each section individually with detailed extraction
        3. Combine results into final SRS document
        """
        print("Starting sectional SRS generation process...")
        
        # Read input from text file if input is too short (likely reading from PDF failed)
        if len(text.strip()) < 100:
            print("WARNING: Input text is too short. Attempting to read from dineout.txt...")
            try:
                with open('input/dineout.txt', 'r', encoding='utf-8') as file:
                    text = file.read()
                print(f"Successfully read {len(text)} characters from backup text file.")
            except Exception as e:
                print(f"ERROR reading backup text file: {str(e)}")
        
        # Define the SRS sections to extract, prioritizing those critical for entity extraction
        sections = [
            {
                "name": "system_info",
                "focus": "system name, purpose, scope, description and boundaries",
                "output_keys": ["system"],
                "importance": "critical",
                "extraction_notes": "Provides the system boundary for the use case diagram"
            },
            {
                "name": "actors_and_users",
                "focus": "stakeholders, users, roles, external systems and all entities that interact with the system",
                "output_keys": ["stakeholders", "users"],
                "importance": "critical",
                "extraction_notes": "These will become actors in the use case diagram"
            },
            {
                "name": "functional_requirements",
                "focus": "detailed functional requirements with specific actions, behaviors and capabilities the system must perform",
                "output_keys": ["functional_requirements"],
                "importance": "critical",
                "extraction_notes": "These will be transformed into use cases"
            },
            {
                "name": "interfaces",
                "focus": "user interfaces, system interfaces, hardware interfaces and communication interfaces",
                "output_keys": ["interfaces"],
                "importance": "high",
                "extraction_notes": "Helps identify relationships between actors and use cases"
            },
            {
                "name": "non_functional_requirements",
                "focus": "performance, security, reliability requirements that may influence system behavior",
                "output_keys": ["non_functional_requirements"],
                "importance": "medium",
                "extraction_notes": "Can provide context for use case constraints"
            },
            {
                "name": "constraints_assumptions",
                "focus": "business rules, technical constraints, assumptions about system operation",
                "output_keys": ["constraints", "assumptions"],
                "importance": "medium",
                "extraction_notes": "May indicate include/extend relationships in use cases"
            }
        ]
        
        # Initialize the complete SRS
        complete_srs = {}
        
        # Track success of extraction for each section
        extraction_success = {}
        
        # Process each section individually, prioritizing critical sections
        for section in sections:
            print(f"Processing '{section['name']}' section (importance: {section['importance']})...")
            
            # Use higher max_tokens and lower temperature for critical sections
            max_tokens = 3000 if section['importance'] == "critical" else 2000
            temperature = 0.1 if section['importance'] == "critical" else 0.2
            
            # Give more tokens specifically for functional requirements extraction
            if section['name'] == "functional_requirements":
                max_tokens = 4000  # Increased token limit specifically for functional requirements
                
            section_data = self._process_srs_section(
                text, 
                section["name"], 
                section["focus"],
                max_tokens=max_tokens,
                temperature=temperature,
                extraction_notes=section["extraction_notes"]
            )
            
            # Extract relevant keys from the section response
            section_success = False
            for key in section["output_keys"]:
                if key in section_data and section_data[key]:  # Check if key exists and is not empty
                    complete_srs[key] = section_data[key]
                    section_success = True
            
            extraction_success[section["name"]] = section_success
        
        # Validate the final SRS and check all required keys
        all_required_keys = ["system", "stakeholders", "functional_requirements","constraints"]
        missing_keys = [key for key in all_required_keys if key not in complete_srs]
        
        if missing_keys:
            print(f"WARNING: Missing required keys in final SRS: {missing_keys}. Found: {list(complete_srs.keys())}.")
            
            # Get total length of text for more aggressive recovery
            text_length = len(text)
            max_segment_size = min(16000, text_length // 1.5)  # Increased segment size from 8000 to 16000
            
            # Try to recover with more focused prompts and chunking
            for key in missing_keys:
                print(f"Attempting to recover missing '{key}' section (CRITICAL)...")
                
                # Find which section contains this key
                for section in sections:
                    if key in section["output_keys"]:
                        # Try multiple strategies for recovery
                        recovery_successful = False
                        
                        # Strategy 1: Use text chunks for targeted extraction
                        chunks = [text[i:i+max_segment_size] for i in range(0, text_length, max_segment_size)]  # Non-overlapping chunks (removed //2)
                        
                        for i, chunk in enumerate(chunks):
                            print(f"Analyzing chunk {i+1}/{len(chunks)} for '{key}'...")
                            
                            # Try with higher temperature and more tokens for recovery
                            section_data = self._process_srs_section(
                                chunk,
                                section["name"],
                                section["focus"],
                                is_recovery=True,
                                max_tokens=5000,  # Significantly increased token limit for recovery
                                temperature=0.4,
                                extraction_notes=section["extraction_notes"]
                            )
                            
                            if key in section_data and section_data[key] and len(section_data[key]) > 0:
                                complete_srs[key] = section_data[key]
                                recovery_successful = True
                                print(f"Successfully recovered '{key}' from chunk {i+1}")
                                break
                        
                        # Strategy 2: Try a different extraction approach if chunking failed
                        if not recovery_successful and key == "functional_requirements":
                            print("Trying alternative extraction approach for functional requirements...")
                            
                            # Look for action-oriented statements in the text
                            fr_data = self._extract_fr_from_text_directly(text)
                            if fr_data and len(fr_data) > 0:
                                complete_srs["functional_requirements"] = fr_data
                                recovery_successful = True
                                print(f"Successfully extracted {len(fr_data)} functional requirements using alternative approach")
                            
                        
                                
        # Fix FR IDs if present
        if "functional_requirements" in complete_srs:
            complete_srs = self._fix_fr_ids(complete_srs)
        
        print(f"Completed sectional SRS generation with {len(complete_srs.get('functional_requirements', []))} functional requirements")
        return complete_srs

    def _process_srs_section(self, text: str, section_name: str, section_focus: str, 
                          is_recovery: bool = False, max_tokens: int = 1500, 
                          temperature: float = 0.1, extraction_notes: str = "") -> Dict:
        """
        Process a specific section of the SRS using focused prompting.
        
        Args:
            text (str): The input text to analyze
            section_name (str): Name of the section being processed
            section_focus (str): Description of what to focus on in this section
            is_recovery (bool): Whether this is a recovery attempt for a missing section
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature setting for generation
            extraction_notes (str): Notes about how this section relates to entity extraction
            
        Returns:
            Dict: The processed section data
        """
        # Set higher temperature for recovery attempts to get more varied results
        if is_recovery:
            temperature += 0.2  # Increase temperature for recovery
        
        # Create a prompt focused on the specific section
        prompt = f"""
        You are an expert requirements analyst specializing in Software Requirements Specification.
        
        TASK: Extract ONLY the {section_focus} from the provided input text.
        
        CRITICAL INSTRUCTIONS:
        1. Focus EXCLUSIVELY on {section_focus} - ignore other aspects of SRS documentation
        2. Be THOROUGH and DETAILED for this specific section
        3. Include ALL relevant details from the input text related to {section_focus}
        4. Be specific with clear, actionable information
        5. Only include information that is explicitly mentioned or strongly implied
        6. IMPORTANT FOR UML: {extraction_notes}
        7. YOU MUST RESPOND WITH A COMPLETE JSON OBJECT CONTAINING ALL EXTRACTED INFORMATION
        8. YOUR RESPONSE MUST NOT BE EMPTY OR SHORT - IT MUST INCLUDE ALL DETAILS FOUND
        9. IF NO INFORMATION IS FOUND, PROVIDE AN EXPLANATION WITHIN THE JSON STRUCTURE
        
        The output MUST be a valid JSON object with ONLY the following keys related to {section_focus}:
        """
        
        # Add the appropriate schema section based on what we're extracting
        if "system" in section_focus:
            prompt += """
            {
                "system": {
                    "name": "Specific system name",
                    "description": "Detailed system description",
                    "purpose": "Clear statement of purpose",
                    "scope": "Specific system boundaries and scope"
                }
            }
            """
        elif "stakeholders" in section_focus or "users" in section_focus or "actors" in section_focus:
            prompt += """
            {
                "stakeholders": [
                    {
                        "name": "Specific stakeholder name/role",
                        "role": "Detailed stakeholder role",
                        "needs": "Specific needs and expectations",
                        "interactions": "How they interact with the system"
                    }
                ],
                "users": [
                    {
                        "type": "Specific user type/role",
                        "description": "Detailed description of this user",
                        "expectations": "Specific expectations from the system",
                        "actions": "Key actions this user performs with the system"
                    }
                ]
            }
            """
        elif "functional_requirements" in section_focus:
            prompt += """
            {
                "functional_requirements": [
                    {
                        "id": "FR-001",
                        "description": "Detailed, specific requirement description with action verbs",
                        "priority": "High/Medium/Low",
                        "source": "Source of requirement if mentioned",
                        "actors_involved": "Which users/roles interact with this functionality",
                        "triggers": "What initiates this functionality"
                    }
                ]
            }
            """
        elif "non_functional_requirements" in section_focus:
            prompt += """
            {
                "non_functional_requirements": [
                    {
                        "category": "Performance/Security/Reliability/etc.",
                        "description": "Detailed requirement description",
                        "importance": "Critical/High/Medium/Low",
                        "verification_method": "How this requirement would be verified"
                    }
                ]
            }
            """
        elif "interfaces" in section_focus:
            prompt += """
            {
                "interfaces": [
                    {
                        "type": "User/Hardware/Software/Communication",
                        "description": "Detailed interface description",
                        "connecting_entities": "What entities connect through this interface",
                        "data_exchanged": "What information flows through this interface"
                    }
                ]
            }
            """
        elif "constraints" in section_focus or "assumptions" in section_focus:
            prompt += """
            {
                "constraints": [
                    "Specific technical, business, or regulatory constraints"
                ],
                "assumptions": [
                    "Specific assumptions made about the system or environment"
                ]
            }
            """
            
        # Add the input text
        prompt += f"""
        
        Input Text:
        ```
        {text}
        ```
        
        IMPORTANT: Return ONLY valid JSON with the essential information for {section_name}.
        Focus on COMPLETE, DETAILED information that will be valuable for UML entity extraction.
        Do not include any explanatory text outside the JSON structure.
        YOUR RESPONSE MUST BE A VALID JSON OBJECT, NOT TEXT.
        """
        
        # Add extra emphasis for recovery attempts
        if is_recovery:
            prompt += f"""
            
            CRITICAL: This is a recovery attempt for missing {section_name} information.
            You MUST extract this information even if it requires reasonable inference
            from the text, while maintaining accuracy and relevance.
            This section is ESSENTIAL for proper UML entity extraction.
            """
        
        # Process with multiple retries
        for attempt in range(self.max_retries):
            try:
                print(f"Processing '{section_name}' (attempt {attempt + 1}/{self.max_retries})...")
                
                # Increase temperature slightly with each retry
                current_temp = temperature + (0.1 * attempt)
                
                response = self._throttled_llm_request(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=current_temp
                )
                
                # Check response length - if too short, retry with higher temperature
                if len(response) < 50:  # Very short response
                    print(f"WARNING: Response too short ({len(response)} chars). Retrying with higher temperature.")
                    continue
                
                # Clean the response - remove any markdown formatting
                response = response.replace('```json', '').replace('```', '').strip()
                
                # Extract JSON
                match = re.search(r'{.*}', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response  # Assume the response is the JSON string if no clear delimiters

                try:
                    section_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in '{section_name}' response: {str(e)}. Attempting to fix common issues.")
                    # Try to fix common issues
                    json_str_fixed = re.sub(r',\s*([\}\]])', r'\1', json_str)  # remove trailing commas
                    try:
                        section_data = json.loads(json_str_fixed)
                        print("INFO: Successfully parsed JSON after attempting fixes.")
                    except json.JSONDecodeError as e_fix:
                        print(f"ERROR: Still invalid JSON after attempting fixes: {str(e_fix)}.")
                        if attempt == self.max_retries - 1:
                            print(f"Failed to parse JSON for '{section_name}' after {self.max_retries} attempts and fixes.")
                            return {}  # Return empty dict for this section, we'll try recovery later
                        continue  # Retry

                # Check if we got empty arrays for critical fields
                if section_name == "functional_requirements" and "functional_requirements" in section_data:
                    if not section_data["functional_requirements"] or len(section_data["functional_requirements"]) == 0:
                        print(f"WARNING: Empty functional_requirements array. Retrying with different approach.")
                        if attempt == self.max_retries - 1:
                            # On last attempt, keep the empty result but log warning
                            print(f"WARNING: Could not extract functional requirements after {self.max_retries} attempts.")
                            return section_data
                        continue  # Try again with higher temperature
                
                return section_data
                    
            except Exception as e:
                print(f"ERROR: Attempt {attempt + 1} for '{section_name}' processing failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    print(f"Failed to process '{section_name}' after {self.max_retries} attempts.")
                    return {}  # Return empty dict for this section, we'll try recovery later
                continue
        
        return {}  # Return empty dict if all attempts fail

    def extract_entities(self, srs: Dict) -> Dict:
        """
        Extract entities for UML using a level-by-level prompting approach:
        1. First level: Extract actors with detailed descriptions
        2. Second level: Extract use cases with clear verb-noun naming
        3. Third level: Extract relationships between actors and use cases
        4. Fourth level: Refine relationships to ensure proper connectivity
        
        Args:
            srs (Dict): Standardized SRS document
            
        Returns:
            Dict: Extracted entities (system, actors, use_cases, relationships)
        """
        print("Starting level-by-level extraction of UML entities...")
        
        # Initialize the result structure
        entities = {
            "system": {},
            "actors": [],
            "use_cases": [],
            "relationships": []
        }
        
        # Extract system information
        system_info = self._extract_system_info(srs)
        entities["system"] = system_info
        
        # Level 1: Extract actors
        actors = self._extract_actors(srs)
        entities["actors"] = actors
        print(f"Level 1: Extracted {len(actors)} actors")
        
        # Level 2: Extract use cases
        use_cases = self._extract_use_cases(srs)
        entities["use_cases"] = use_cases
        print(f"Level 2: Extracted {len(use_cases)} use cases")
        
        # Level 3: Extract primary relationships
        relationships = self._extract_relationships(srs, actors, use_cases)
        entities["relationships"] = relationships
        print(f"Level 3: Extracted {len(relationships)} relationships")
        
        # Level 4: Refine and ensure connectivity
        entities = self._refine_relationships(entities)
        print(f"Level 4: Refined model now has {len(entities['relationships'])} relationships")
        
        return entities
    
    def _extract_system_info(self, srs: Dict) -> Dict:
        """Extract system information from the SRS"""
        if "system" in srs and isinstance(srs["system"], dict):
            return {
                "name": srs["system"].get("name", "System"),
                "description": srs["system"].get("description", srs["system"].get("purpose", ""))
            }
        return {"name": "System", "description": "System extracted from requirements"}
    
    def _extract_actors(self, srs: Dict) -> List[Dict]:
        """
        Extract actors using a focused prompt specifically for actor identification.
        
        Args:
            srs: The standardized SRS document
            
        Returns:
            List of actor dictionaries
        """
        # Create a condensed version of the SRS for the prompt
        srs_text = json.dumps({
            "system": srs.get("system", {}),
            "stakeholders": srs.get("stakeholders", []),
            "functional_requirements": srs.get("functional_requirements", {})
        }, indent=2)
        
        # First try direct extraction from stakeholders
        actors = []
        if "stakeholders" in srs and isinstance(srs["stakeholders"], list):
            for stakeholder in srs["stakeholders"]:
                if isinstance(stakeholder, dict):
                    # Determine actor type based on name/description
                    actor_type = "primary"
                    name = stakeholder.get("name", "")
                    description = stakeholder.get("description", "")
                    
                    # Check if this is likely an external system
                    if any(term in description.lower() for term in ["external", "system", "interface", "third-party", "api"]):
                        actor_type = "external-system"
                        
                    # Check if this is likely a secondary actor
                    if any(term in description.lower() for term in ["support", "secondary", "assist"]):
                        actor_type = "secondary"
                        
                    actors.append({
                        "name": name,
                        "type": actor_type,
                        "description": description
                    })
            
            # If we found actors directly, return them
            if actors:
                print(f"Extracted {len(actors)} actors directly from stakeholders")
                return actors
        
        # If direct extraction failed or returned no actors, use LLM
        prompt = f"""
        FOCUS: ACTOR IDENTIFICATION FOR UML USE CASE DIAGRAM
        
        Analyze this SRS and identify ALL actors who interact with the system.
        
        CRITICAL GUIDELINES FOR ACTOR IDENTIFICATION:
        1. An actor is an entity OUTSIDE the system that interacts WITH the system
        2. Actors can be people, external systems, or hardware devices
        3. Actors should be specific roles (e.g., "Customer" not "Person")
        4. Each actor must have a distinct responsibility or purpose
        5. Include primary actors (who initiate use cases) and secondary actors (who respond)
        6. External systems that exchange data with this system are actors
        7. Time triggers or scheduling systems can be actors
        
        Return a JSON array of actors with this structure:
        [
            {{
                "name": "Actor name (role-based, specific)",
                "type": "primary/secondary/external-system/hardware",
                "description": "Clear description of the actor's role and responsibilities"
            }}
        ]
        
        SRS Document:
        {srs_text}
        
        Return ONLY the JSON array with the actors, nothing else.
        """
        
        for attempt in range(self.max_retries):
            try:
                print(f"Extracting actors (attempt {attempt + 1}/{self.max_retries})...")
                response = self._throttled_llm_request(prompt, max_tokens=1500, temperature=0.1)
                
                # Clean response and extract JSON
                response = response.replace('```json', '').replace('```', '').strip()
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response
                
                try:
                    actors = json.loads(json_str)
                    if not isinstance(actors, list):
                        raise ValueError("Actor extraction did not return a list")
                        
                    print(f"Successfully extracted {len(actors)} actors")
                    return actors
                    
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in actors response: {str(e)}. Response snippet: {json_str[:500]}")
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"Failed to parse JSON for actors after {self.max_retries} attempts")
                    continue
                
            except Exception as e:
                print(f"ERROR: Attempt {attempt + 1} for actor extraction failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to extract actors after {self.max_retries} attempts: {str(e)}")
                continue
        
        # If all attempts fail
        return []
    
    def _extract_use_cases(self, srs: Dict) -> List[Dict]:
        """
        Extract use cases using a focused prompt specifically for use case identification.
        Includes context about previously identified actors.
        
        Args:
            srs: The standardized SRS document
            
        Returns:
            List of use case dictionaries
        """
        # First try direct extraction from functional requirements
        use_cases = []
        
        # Check if we have functional requirements in the correct format
        if "functional_requirements" in srs:
            fr = srs["functional_requirements"]
            
            if isinstance(fr, dict):
                # If functional_requirements is a dictionary with feature names as keys
                for feature_name, feature_data in fr.items():
                    if isinstance(feature_data, dict):
                        # Convert feature name to verb-noun format for use case name
                        use_case_name = self._convert_to_verb_noun(feature_name.replace("_", " "))
                        
                        use_case = {
                            "name": use_case_name,
                            "description": feature_data.get("description", ""),
                            "priority": feature_data.get("priority", "Medium")
                        }
                        use_cases.append(use_case)
            
            # If we found use cases directly, return them
            if use_cases:
                print(f"Extracted {len(use_cases)} use cases directly from functional requirements")
                return use_cases
        
        # If direct extraction failed or found no use cases, use LLM approach
        # Create a condensed version of the SRS focusing on requirements
        fr_text = json.dumps(srs.get("functional_requirements", {}), indent=2)
        
        # Check if we have reasonable functional requirements to work with
        if not srs.get("functional_requirements"):
            print("WARNING: No functional requirements found. Using system and stakeholders info for use case extraction.")
            # Create a more comprehensive context from system and stakeholders
            context = {
                "system": srs.get("system", {}),
                "stakeholders": srs.get("stakeholders", []),
                "general_requirements": "Restaurant management system must handle restaurant operations and customer interactions."
            }
            fr_text = json.dumps(context, indent=2)
        
        prompt = f"""
        FOCUS: USE CASE IDENTIFICATION FOR UML USE CASE DIAGRAM
        
        Analyze these functional requirements and identify ALL use cases for the system.
        
        CRITICAL GUIDELINES FOR USE CASE IDENTIFICATION:
        1. A use case represents a specific interaction between actors and the system
        2. Each use case must provide value to an actor
        3. Use cases MUST be named in verb-noun format (e.g., "Process Payment" not "Payment Processing")
        4. Focus on COMPLETE user goals, not individual steps or UI interactions
        5. A good use case answers "What can actors DO with the system?"
        6. Avoid technical details or implementation specifics
        7. Include a clear description that explains what the use case accomplishes
        8. For a restaurant system, common use cases would include "Place Order", "Process Payment", "Manage Tables", etc.
        
        Return a JSON array of use cases with this structure:
        [
            {{
                "name": "Use case name (verb-noun format)",
                "description": "Clear description of what this use case accomplishes",
                "priority": "High/Medium/Low"
            }}
        ]
        
        Functional Requirements:
        {fr_text}
        
        Return ONLY the JSON array with the use cases, nothing else.
        """
        
        for attempt in range(self.max_retries):
            try:
                print(f"Extracting use cases from LLM (attempt {attempt + 1}/{self.max_retries})...")
                response = self._throttled_llm_request(prompt, max_tokens=2000, temperature=0.1 + (0.1 * attempt))
                
                # Check response length - if too short, try again with different approach
                if len(response) < 20:  # Too short to be valid JSON
                    print(f"WARNING: Response too short: {len(response)} characters. Trying different approach...")
                    continue
                
                # Clean response and extract JSON
                response = response.replace('```json', '').replace('```', '').strip()
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response
                
                try:
                    extracted_use_cases = json.loads(json_str)
                    if not isinstance(extracted_use_cases, list):
                        print("WARNING: Use case extraction did not return a list. Retrying...")
                        continue
                        
                    if len(extracted_use_cases) == 0:
                        print("WARNING: No use cases extracted. Retrying with different approach...")
                        continue
                        
                    # Ensure all use cases have the required format
                    for uc in extracted_use_cases:
                        if "name" in uc and not self._is_verb_noun_format(uc["name"]):
                            uc["name"] = self._convert_to_verb_noun(uc["name"])
                            
                    print(f"Successfully extracted {len(extracted_use_cases)} use cases")
                    use_cases = extracted_use_cases
                    break  # Successful extraction
                    
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in use cases response: {str(e)}. Response snippet: {json_str[:100]}...")
                    if attempt == self.max_retries - 1:
                        print(f"WARNING: Failed to parse JSON for use cases after {self.max_retries} attempts. Using fallback...")
                    continue
                
            except Exception as e:
                print(f"ERROR: Attempt {attempt + 1} for use case extraction failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    print(f"WARNING: Failed to extract use cases after {self.max_retries} attempts. Using fallback...")
                continue
        
        # If all extraction attempts failed, create fallback use cases
        if not use_cases:
            print("Creating fallback use cases from system context...")
            system_name = srs.get("system", {}).get("name", "DineOut System")
            actor_names = [actor.get("name", "User") for actor in srs.get("stakeholders", [])]
            
            # Create generic use cases based on system context
            fallback_use_cases = [
                {
                    "name": "Manage Restaurant Operations",
                    "description": "Allow managers to manage all aspects of restaurant operations",
                    "priority": "High"
                },
                {
                    "name": "Process Customer Orders",
                    "description": "Process and manage customer food orders",
                    "priority": "High"
                },
                {
                    "name": "Manage Reservations",
                    "description": "Handle table reservations for customers",
                    "priority": "Medium"
                },
                {
                    "name": "Process Payments",
                    "description": "Process customer payments for orders",
                    "priority": "High"
                },
                {
                    "name": "Generate Reports",
                    "description": "Generate business reports for management",
                    "priority": "Medium"
                }
            ]
            
            # Add actor-specific use cases if actors are available
            for actor in actor_names:
                if "manager" in actor.lower() or "admin" in actor.lower():
                    fallback_use_cases.append({
                        "name": f"Manage Staff",
                        "description": f"Allow {actor} to manage restaurant staff",
                        "priority": "Medium"
                    })
                elif "customer" in actor.lower() or "client" in actor.lower():
                    fallback_use_cases.append({
                        "name": f"Place Order",
                        "description": f"Allow {actor} to place food orders",
                        "priority": "High"
                    })
                    fallback_use_cases.append({
                        "name": f"Make Reservation",
                        "description": f"Allow {actor} to make table reservations",
                        "priority": "Medium"
                    })
            
            print(f"Created {len(fallback_use_cases)} fallback use cases")
            return fallback_use_cases
        
        return use_cases
    
    def _extract_relationships(self, srs: Dict, actors: List[Dict], use_cases: List[Dict]) -> List[Dict]:
        """
        Extract relationships using a focused prompt with knowledge of actors and use cases.
        
        Args:
            srs: The standardized SRS document
            actors: Previously extracted actors
            use_cases: Previously extracted use cases
            
        Returns:
            List of relationship dictionaries
        """
        # Create a context for the prompt with actors and use cases
        actors_text = json.dumps(actors, indent=2)
        use_cases_text = json.dumps(use_cases, indent=2)
        
        prompt = f"""
        FOCUS: RELATIONSHIP IDENTIFICATION FOR UML USE CASE DIAGRAM
        
        You have these actors and use cases extracted from an SRS.
        Now identify ALL relationships between them.
        
        CRITICAL GUIDELINES FOR RELATIONSHIP IDENTIFICATION:
        1. Association: Direct interaction between an actor and a use case
           - Every actor must connect to at least one use case
           - Every use case must be associated with at least one actor
           
        2. Include: One use case includes functionality of another use case
           - Format: <base use case> includes <included use case>
           - The included use case is ALWAYS performed as part of the base use case
           - Example: "Process Order" includes "Validate Payment"
           
        3. Extend: One use case may optionally extend another use case
           - Format: <extending use case> extends <base use case>
           - The extending use case adds optional behavior to the base use case
           - Example: "Apply Discount" extends "Process Order"
        
        Return a JSON array of relationships with this structure:
        [
            {{
                "source": "Actor or use case name (for include/extend)",
                "target": "Use case name",
                "type": "association/include/extend",
                "description": "Optional description of the relationship"
            }}
        ]
        
        Actors:
        {actors_text}
        
        Use Cases:
        {use_cases_text}
        
        Return ONLY the JSON array with the relationships, nothing else.
        """
        
        for attempt in range(self.max_retries):
            try:
                print(f"Extracting relationships (attempt {attempt + 1}/{self.max_retries})...")
                response = self._throttled_llm_request(prompt, max_tokens=2000, temperature=0.1)
                
                # Clean response and extract JSON
                response = response.replace('```json', '').replace('```', '').strip()
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response
                
                try:
                    relationships = json.loads(json_str)
                    if not isinstance(relationships, list):
                        raise ValueError("Relationship extraction did not return a list")
                        
                    # Support both source/target and from/to formats
                    for rel in relationships:
                        if "from" in rel and "to" in rel:
                            rel["source"] = rel.pop("from")
                            rel["target"] = rel.pop("to")
                            
                    print(f"Successfully extracted {len(relationships)} relationships")
                    return relationships
                    
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in relationships response: {str(e)}. Response snippet: {json_str[:100]}...")
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"Failed to parse JSON for relationships after {self.max_retries} attempts")
                    continue
                
            except Exception as e:
                print(f"ERROR: Attempt {attempt + 1} for relationship extraction failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to extract relationships after {self.max_retries} attempts: {str(e)}")
                continue
        
        # If all attempts fail
        return []
    
    def _refine_relationships(self, entities: Dict) -> Dict:
        """
        Final refinement step to ensure proper connectivity and relationship types.
        
        Args:
            entities: The extracted entities
            
        Returns:
            Dict: Refined entities with improved connectivity
        """
        actors = entities.get("actors", [])
        use_cases = entities.get("use_cases", [])
        relationships = entities.get("relationships", [])
        
        # Convert actor and use case names to sets for faster lookup
        actor_names = {a["name"] for a in actors if isinstance(a, dict) and "name" in a}
        usecase_names = {uc["name"] for uc in use_cases if isinstance(uc, dict) and "name" in uc}
        
        # Track which actors and use cases are connected
        connected_actors = set()
        connected_use_cases = set()
        
        # Check existing relationships and identify connectivity
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
                
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "").lower()
            
            # Check if source is an actor
            if source in actor_names:
                connected_actors.add(source)
                
                # For actor-use case associations, validate the target is a use case
                if rel_type == "association" and target in usecase_names:
                    connected_use_cases.add(target)
            
            # For include/extend relationships, add both source and target to connected use cases
            if rel_type in ["include", "extend"] and source in usecase_names and target in usecase_names:
                connected_use_cases.add(source)
                connected_use_cases.add(target)
        
        # Prompt LLM to analyze any connectivity gaps
        if len(actor_names - connected_actors) > 0 or len(usecase_names - connected_use_cases) > 0:
            unconnected_actors = list(actor_names - connected_actors)
            unconnected_use_cases = list(usecase_names - connected_use_cases)
            
            prompt = f"""
            FOCUS: FIXING CONNECTIVITY GAPS IN UML DIAGRAM
            
            There are unconnected elements in the UML diagram:
            
            Unconnected Actors: {json.dumps(unconnected_actors)}
            Unconnected Use Cases: {json.dumps(unconnected_use_cases)}
            
            Create appropriate relationships to connect these elements:
            
            1. Each actor should connect to at least one use case
            2. Each use case should be connected to at least one actor or other use case
            3. ONLY suggest relationships that make logical sense based on actor and use case descriptions
            
            Return a JSON array of ADDITIONAL relationships with this structure:
            [
                {{
                    "source": "Actor or use case name",
                    "target": "Use case name",
                    "type": "association/include/extend",
                    "description": "Explanation of why this relationship exists"
                }}
            ]
            
            All Actors:
            {json.dumps([a["name"] for a in actors], indent=2)}
            
            All Use Cases:
            {json.dumps([uc["name"] for uc in use_cases], indent=2)}
            
            Return ONLY the JSON array with the new relationships.
            """
            
            try:
                print("Attempting to fix connectivity gaps in the UML diagram...")
                response = self._throttled_llm_request(prompt, max_tokens=1000, temperature=0.2)
                
                # Clean response and extract JSON
                response = response.replace('\'\'\'json', '').replace('\'\'\'', '').strip()
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response
                
                try:
                    additional_relationships = json.loads(json_str)
                    if isinstance(additional_relationships, list):
                        # Add the new relationships to the existing set
                        relationships.extend(additional_relationships)
                        print(f"Added {len(additional_relationships)} relationships to fix connectivity gaps")
                except Exception as e:
                    print(f"WARNING: Failed to parse additional relationships: {str(e)}")
            except Exception as e:
                print(f"WARNING: Failed to generate additional relationships: {str(e)}")
        
        # Update entities with refined relationships
        entities["relationships"] = relationships
        
        return entities

    def _is_verb_noun_format(self, text: str) -> bool:
        """Check if text is in verb-noun format"""
        # Simple regex to check if text starts with a verb
        verb_pattern = r'^(?:get|set|create|update|delete|manage|process|validate|check|verify|add|remove|edit|view|list|search|generate|calculate|send|receive|login|logout|register|upload|download|modify|access|retrieve|store|save)'
        return bool(re.match(verb_pattern, text.lower()))

    def _convert_to_verb_noun(self, text: str) -> str:
        """Convert a phrase to verb-noun format if possible"""
        text = text.strip()
        
        # Already in verb-noun format
        if self._is_verb_noun_format(text):
            return text
            
        # Handle special cases
        lower_text = text.lower()
        if 'registration' in lower_text:
            return 'Register ' + text.replace('Registration', '').replace('registration', '').strip()
        elif 'management' in lower_text:
            return 'Manage ' + text.replace('Management', '').replace('management', '').strip()
        elif 'creation' in lower_text:
            return 'Create ' + text.replace('Creation', '').replace('creation', '').strip()
        elif 'deletion' in lower_text:
            return 'Delete ' + text.replace('Deletion', '').replace('deletion', '').strip()
        elif 'modification' in lower_text:
            return 'Modify ' + text.replace('Modification', '').replace('modification', '').strip()
        elif 'viewing' in lower_text:
            return 'View ' + text.replace('Viewing', '').replace('viewing', '').strip()
        
        # Default case - add generic verb based on context
        if 'add' in lower_text or 'create' in lower_text or 'new' in lower_text:
            return 'Create ' + text
        elif 'edit' in lower_text or 'update' in lower_text or 'modify' in lower_text:
            return 'Update ' + text
        elif 'delete' in lower_text or 'remove' in lower_text:
            return 'Delete ' + text
        elif 'view' in lower_text or 'display' in lower_text or 'show' in lower_text:
            return 'View ' + text
        elif 'manage' in lower_text:
            return 'Manage ' + text
        else:
            # Default to "Manage" if no clear verb is detected
            return 'Manage ' + text

    def _extract_use_case_name(self, text: str) -> str:
        """Extract a use case name from text"""
        # Check for verb-phrase patterns
        verb_patterns = [
            r'((?:should|must|will|can|may)\s+(?:be\s+able\s+to\s+)?)((?:get|set|create|update|delete|manage|process|validate|check|verify|add|remove|edit|view|list|search|generate|calculate|send|receive|login|logout|register|upload|download|modify|access|retrieve|store|save)\s+[\w\s]+)',
            r'(The\s+system\s+(?:should|must|will|can|may)\s+(?:allow|enable|let|permit)\s+[\w\s]+\s+to\s+)((?:get|set|create|update|delete|manage|process|validate|check|verify|add|remove|edit|view|list|search|generate|calculate|send|receive|login|logout|register|upload|download|modify|access|retrieve|store|save)\s+[\w\s]+)'
        ]
        
        for pattern in verb_patterns:
            match = re.search(pattern, text)
            if match:
                # Extract just the action part
                action = match.group(2).strip()
                # Capitalize first letter
                return action[0].upper() + action[1:]
        
        # Simpler extraction if patterns don't match
        # Look for the first verb in the text
        words = text.split()
        for i, word in enumerate(words):
            word = word.lower().strip('.,;:()"\'')
            if word in ['get', 'set', 'create', 'update', 'delete', 'manage', 'process', 'validate', 
                       'check', 'verify', 'add', 'remove', 'edit', 'view', 'list', 'search']:
                # Take the verb and up to 3 following words
                end_idx = min(i + 4, len(words))
                phrase = ' '.join(words[i:end_idx])
                # Clean up the phrase
                phrase = re.sub(r'[^\w\s]', '', phrase)
                return phrase[0].upper() + phrase[1:]
        
        # If all else fails, create a name from the first part of the text
        if text:
            # Get first 30 chars or first sentence, whichever is shorter
            first_part = text.split('.')[0][:30].strip()
            return self._convert_to_verb_noun(first_part)
            
        return ""

    def _find_best_use_case_match(self, actor_name: str, use_cases: List[Dict]) -> str:
        """Find the most relevant use case for an actor based on name similarity"""
        actor_lower = actor_name.lower()
        
        # Check for direct word matches in name or description
        for uc in use_cases:
            uc_name = uc.get("name", "").lower()
            uc_desc = uc.get("description", "").lower()
            
            # Check if actor name appears in use case name or description
            if actor_lower in uc_name or actor_lower in uc_desc:
                return uc.get("name")
        
        # If no direct match, use common patterns for specific actors
        actor_patterns = {
            "user": ["login", "register", "view", "search"],
            "admin": ["manage", "configure", "set up", "administrate"],
            "customer": ["order", "purchase", "buy", "cancel", "feedback"],
            "manager": ["review", "approve", "assign", "report"],
            "system": ["process", "calculate", "generate", "validate"]
        }
        
        for pattern, verbs in actor_patterns.items():
            if pattern in actor_lower:
                for uc in use_cases:
                    uc_name = uc.get("name", "").lower()
                    if any(verb in uc_name for verb in verbs):
                        return uc.get("name")
        
        # Default to first use case if no better match
        return use_cases[0].get("name")

    def _find_best_actor_match(self, use_case_name: str, actors: List[Dict]) -> str:
        """Find the most relevant actor for a use case based on name similarity"""
        uc_lower = use_case_name.lower()
        
        # Common use case verbs and their likely actor associations
        verb_actor_map = {
            "manage": ["admin", "manager", "supervisor"],
            "create": ["admin", "user"],
            "view": ["user", "customer", "client"],
            "edit": ["admin", "user", "manager"],
            "delete": ["admin"],
            "register": ["user", "customer", "client"],
            "login": ["user", "admin", "customer"],
            "search": ["user", "customer", "client"],
            "order": ["customer", "client"],
            "purchase": ["customer", "client"],
            "report": ["manager", "admin"],
            "assign": ["manager", "admin"],
            "configure": ["admin"],
            "approve": ["manager", "admin"]
        }
        
        # Find verb in use case
        for verb, likely_actors in verb_actor_map.items():
            if verb in uc_lower:
                # Check if we have any of the likely actors
                for actor in actors:
                    actor_name = actor.get("name", "").lower()
                    if any(likely in actor_name for likely in likely_actors):
                        return actor.get("name")
        
        # Default: if use case is administrative, assign to admin/manager type
        if any(term in uc_lower for term in ["admin", "manage", "config", "setup", "report"]):
            for actor in actors:
                actor_name = actor.get("name", "").lower()
                if any(term in actor_name for term in ["admin", "manager", "supervisor"]):
                    return actor.get("name")
        
        # Otherwise, assign to first actor (usually a primary user)
        return actors[0].get("name")

    def _is_complex_use_case(self, use_case: Dict) -> bool:
        """Check if a use case is complex enough to be a base for include/extend"""
        name = use_case.get("name", "").lower()
        desc = use_case.get("description", "").lower()
        
        # Complex use cases often have certain verbs or are described as multi-step
        complex_indicators = [
            "manage", "process", "handle", "coordinate", "multiple", "steps", 
            "workflow", "several", "various", "different"
        ]
        
        return any(indicator in name or indicator in desc for indicator in complex_indicators)

    def _is_includable_use_case(self, use_case: Dict) -> bool:
        """Check if a use case is a good candidate to be included by others"""
        name = use_case.get("name", "").lower()
        desc = use_case.get("description", "").lower()
        
        # Include relationships often involve validation, authentication, common subtasks
        include_indicators = [
            "validate", "verify", "check", "authenticate", "login", "select", 
            "common", "always", "required", "necessary", "must", "payment"
        ]
        
        return any(indicator in name or indicator in desc for indicator in include_indicators)

    def _extract_fr_from_text_directly(self, text: str) -> List[Dict]:
        """
        Extract functional requirements directly from text by looking for action-oriented statements.
        Used as a fallback when structured extraction fails.
        """
        # First try to extract using a different LLM prompt focused on action statements
        prompt = f"""
        TASK: Extract FUNCTIONAL REQUIREMENTS from the following text.

        CRITICAL INSTRUCTIONS:
        - Look for statements that describe what the system MUST/SHOULD/WILL do
        - Focus on action verbs and system capabilities
        - Each requirement must describe a specific functionality
        - Format as a list of distinct functional requirements
        - Do NOT include non-functional aspects like performance or security
        - Include EVERY functional capability mentioned in the text
        - For restaurant management system, look for ordering, reservation, payment, management capabilities

        RESPONSE FORMAT:
        Return the requirements in JSON format as an array of objects:
        [
            {{
                "id": "FR-001",  
                "description": "The complete requirement statement",
                "priority": "High/Medium/Low",
                "actors_involved": "User role involved with this requirement"
            }},
            ...  
        ]

        TEXT:
        ```
        {text[:12000]}  
        ```

        Return ONLY the JSON array, nothing else.
        """
        
        try:
            response = self._throttled_llm_request(
                prompt, 
                max_tokens=6000,  # Very high token limit for this critical extraction
                temperature=0.3
            )
            
            # Parse the JSON response
            response = response.replace('```json', '').replace('```', '').strip()
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = response
                
            try:
                requirements = json.loads(json_str)
                if isinstance(requirements, list) and len(requirements) > 0:
                    return requirements
            except json.JSONDecodeError:
                print("Failed to parse functional requirements from direct extraction")
        except Exception as e:
            print(f"Error in direct FR extraction: {str(e)}")
            
        # If the above failed, return an empty list
        return []
    
