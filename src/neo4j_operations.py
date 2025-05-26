from neo4j import GraphDatabase
from neo4j.time import DateTime # Import DateTime for type checking
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Any
import json
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class Neo4jOperations:
    def __init__(self, uri: str, user: str, password: str):
        logger.info(f"Connecting to Neo4j at {uri}")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._setup_constraints()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
    def _setup_constraints(self):
        """Set up Neo4j constraints."""
        try:
            with self.driver.session() as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:UseCase) REQUIRE u.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:System) REQUIRE s.name IS UNIQUE")
                logger.info("Successfully set up Neo4j constraints")
        except Exception as e:
            logger.error(f"Failed to set up constraints: {str(e)}")
            raise
            
    def store_use_case_info(self, use_case_info: Dict[str, Any]):
        """Store use case diagram information in the graph."""
        try:
            with self.driver.session() as session:
                # Store system
                session.run("""
                    MERGE (s:System {name: $name})
                    SET s.description = $description
                """, use_case_info['system'])
                
                # Store actors
                for actor in use_case_info['actors']:
                    session.run("""
                        MERGE (a:Actor {name: $name})
                        SET a.type = $type,
                            a.description = $description
                    """, actor)
                
                # Store use cases
                for use_case in use_case_info['use_cases']:
                    session.run("""
                        MERGE (u:UseCase {name: $name})
                        SET u.description = $description,
                            u.priority = $priority,
                            u.preconditions = $preconditions,
                            u.postconditions = $postconditions
                    """, use_case)
                
                # Store relationships
                for rel in use_case_info['relationships']:
                    if rel['type'] == 'association':
                        session.run("""
                            MATCH (a:Actor {name: $from})
                            MATCH (u:UseCase {name: $to})
                            MERGE (a)-[r:ASSOCIATES_WITH]->(u)
                            SET r.description = $description
                        """, rel)
                    elif rel['type'] == 'include':
                        session.run("""
                            MATCH (u1:UseCase {name: $from})
                            MATCH (u2:UseCase {name: $to})
                            MERGE (u1)-[r:INCLUDES]->(u2)
                            SET r.description = $description
                        """, rel)
                    elif rel['type'] == 'extend':
                        session.run("""
                            MATCH (u1:UseCase {name: $from})
                            MATCH (u2:UseCase {name: $to})
                            MERGE (u1)-[r:EXTENDS]->(u2)
                            SET r.description = $description
                        """, rel)
                
                logger.info("Successfully stored use case information in Neo4j")
                
        except Exception as e:
            logger.error(f"Failed to store use case information: {str(e)}")
            raise
            
    def get_use_case_info(self) -> Dict[str, Any]:
        """Get all use case diagram information from the graph."""
        try:
            with self.driver.session() as session:
                # Get system
                system = session.run("""
                    MATCH (s:System)
                    RETURN s.name as name, s.description as description
                """).data()
                
                # Get actors
                actors = session.run("""
                    MATCH (a:Actor)
                    RETURN a.name as name, a.type as type, a.description as description
                """).data()
                
                # Get use cases
                use_cases = session.run("""
                    MATCH (u:UseCase)
                    RETURN u.name as name, u.description as description,
                           u.priority as priority, u.preconditions as preconditions,
                           u.postconditions as postconditions
                """).data()
                
                # Get relationships
                relationships = session.run("""
                    MATCH (from)-[r]->(to)
                    WHERE (from:Actor OR from:UseCase) AND to:UseCase
                    RETURN from.name as from, to.name as to, type(r) as type,
                           r.description as description
                """).data()
                
                return {
                    "system": system[0] if system else None,
                    "actors": actors,
                    "use_cases": use_cases,
                    "relationships": relationships
                }
                
        except Exception as e:
            logger.error(f"Failed to get use case information: {str(e)}")
            raise
            
    def close(self):
        """Close the Neo4j connection."""
        try:
            self.driver.close()
            logger.info("Successfully closed Neo4j connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
            raise

    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")

    def create_constraints(self):
        """Create unique constraints to prevent duplicates"""
        with self.driver.session() as session:
            # Create constraints for unique names
            session.run("CREATE CONSTRAINT actor_name IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT usecase_name IF NOT EXISTS FOR (u:UseCase) REQUIRE u.name IS UNIQUE")
            session.run("CREATE CONSTRAINT system_name IF NOT EXISTS FOR (s:System) REQUIRE s.name IS UNIQUE")
            logger.info("Constraints created")

    def merge_actor(self, actor_data: Dict) -> str:
        """Merge actor node (create if not exists)"""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (a:Actor {name: $name})
                ON CREATE SET a.description = $description, a.type = $type
                ON MATCH SET a.description = $description, a.type = $type
                RETURN a.name
                """,
                actor_data
            )
            return result.single()[0]

    def merge_use_case(self, usecase_data: Dict) -> str:
        """Merge use case node (create if not exists)"""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (u:UseCase {name: $name})
                ON CREATE SET u.description = $description, u.preconditions = $preconditions,
                             u.postconditions = $postconditions, u.priority = $priority
                ON MATCH SET u.description = $description, u.preconditions = $preconditions,
                            u.postconditions = $postconditions, u.priority = $priority
                RETURN u.name
                """,
                usecase_data
            )
            return result.single()[0]

    def merge_system(self, system_data: Dict) -> str:
        """Merge system node (create if not exists)"""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (s:System {name: $name})
                ON CREATE SET s.description = $description
                ON MATCH SET s.description = $description
                RETURN s.name
                """,
                system_data
            )
            return result.single()[0]

    def create_relationship(self, from_node: str, to_node: str, relationship_type: str, properties: Dict = None):
        """Create relationship between nodes"""
        with self.driver.session() as session:
            query = f"""
            MATCH (a), (b)
            WHERE a.name = $from_name AND b.name = $to_name
            MERGE (a)-[r:{relationship_type}]->(b)
            """
            if properties:
                query += " SET r += $properties"
            session.run(query, {
                "from_name": from_node,
                "to_name": to_node,
                "properties": properties or {}
            })

    def get_related_entities(self, entity_name: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """Get all related entities for a given node"""
        with self.driver.session() as session:
            query = """
            MATCH (n {name: $name})-[r]->(related)
            WHERE $rel_type IS NULL OR type(r) = $rel_type
            RETURN related, type(r) as relationship_type, r
            """
            result = session.run(query, {"name": entity_name, "rel_type": relationship_type})
            return [dict(record) for record in result]

    def get_entity_context(self, entity_name: str, depth: int = 2) -> Dict:
        """Get context for an entity including its relationships up to specified depth"""
        with self.driver.session() as session:
            query = f"""
            MATCH path = (n {{name: $name}})-[*1..{depth}]-(related)
            RETURN path
            """
            result = session.run(query, {"name": entity_name})
            return [dict(record) for record in result]

    def create_actor_node(self, actor: Dict[str, Any]) -> None:
        """Create an actor node in Neo4j."""
        try:
            with self.driver.session() as session:
                query = """
                CREATE (a:Actor {
                    name: $name,
                    description: $description,
                    type: $type
                })
                """
                
                params = {
                    "name": actor["name"],
                    "description": actor.get("description", ""),
                    "type": actor.get("type", "primary")
                }
                
                session.run(query, params)
                logger.info(f"Created actor node: {actor['name']}")
                
        except Exception as e:
            logger.error(f"Error creating actor node: {str(e)}")
            raise

    def store_analysis_results(self, analysis_result: Dict[str, Any]) -> bool:
        """Store analysis results (actors, use cases, relationships) in Neo4j."""
        try:
            with self.driver.session() as session:
                # Print all input data first for debugging
                print("===== ANALYSIS RESULT DATA =====")
                all_usecases = analysis_result.get('use_cases', [])
                print(f"Total use cases in input: {len(all_usecases)}")
                for i, uc in enumerate(all_usecases):
                    if isinstance(uc, dict):
                        print(f"UC{i+1}: {uc.get('name')} - Type: {type(uc)}")
                    else:
                        print(f"UC{i+1}: {uc} - Type: {type(uc)}")
                print("=============================")
                
                # Store System (if any)
                system_info = analysis_result.get('system')
                if system_info and isinstance(system_info, dict) and system_info.get('name'):
                    session.run("""
                        MERGE (s:System {name: $name})
                        ON CREATE SET s.description = $description, s.created_at = datetime()
                        ON MATCH SET s.description = $description, s.updated_at = datetime()
                    """, name=system_info.get('name'), description=system_info.get('description', ''))

                # Store Actors
                stored_actors = []
                for actor in analysis_result.get('actors', []):
                    if isinstance(actor, dict) and actor.get('name'): # Ensure actor is a dict with a name
                        actor_name = actor.get('name')
                        session.run("""
                            MERGE (a:Actor {name: $name})
                            ON CREATE SET a.description = $description, a.type = $type, a.created_at = datetime()
                            ON MATCH SET a.description = $description, a.type = $type, a.updated_at = datetime()
                        """, name=actor_name, description=actor.get('description', ''), type=actor.get('type', 'Primary'))
                        stored_actors.append(actor_name)
                    elif isinstance(actor, str): # Handle if actor is just a name string
                         session.run("""
                            MERGE (a:Actor {name: $name})
                            ON CREATE SET a.created_at = datetime()
                            ON MATCH SET a.updated_at = datetime()
                        """, name=actor)
                         stored_actors.append(actor)
                
                print(f"Stored {len(stored_actors)} actors in Neo4j: {stored_actors}")

                # Store Use Cases
                stored_usecases = []
                skipped_usecases = []
                
                for i, use_case in enumerate(analysis_result.get('use_cases', [])):
                    try:
                        print(f"Processing use case #{i+1}: {use_case}")
                        
                        if isinstance(use_case, dict) and use_case.get('name'):
                            usecase_name = use_case.get('name')
                            
                            # Check for empty or problematic names
                            if not usecase_name.strip():
                                print(f"WARNING: Skipping use case with empty name: {use_case}")
                                skipped_usecases.append(use_case)
                                continue
                                
                            print(f"Creating use case node: {usecase_name}")
                            result = session.run("""
                                MERGE (u:UseCase {name: $name})
                                ON CREATE SET u.description = $description, 
                                              u.preconditions = $preconditions, 
                                              u.postconditions = $postconditions, 
                                              u.priority = $priority,
                                              u.created_at = datetime()
                                ON MATCH SET u.description = $description, 
                                             u.preconditions = $preconditions, 
                                             u.postconditions = $postconditions, 
                                             u.priority = $priority,
                                             u.updated_at = datetime()
                                RETURN u.name
                            """, 
                            name=usecase_name, 
                            description=use_case.get('description', ''),
                            preconditions=use_case.get('preconditions', []),
                            postconditions=use_case.get('postconditions', []),
                            priority=use_case.get('priority', 'Medium'))
                            
                            # Check if node was created successfully
                            if result.peek():
                                stored_usecases.append(usecase_name)
                                print(f"Successfully created use case: {usecase_name}")
                            else:
                                print(f"WARNING: Failed to create use case: {usecase_name}")
                                skipped_usecases.append(use_case)
                        elif isinstance(use_case, str): # Handle if use case is just a name string
                            # Check for empty strings
                            if not use_case.strip():
                                print(f"WARNING: Skipping empty use case name")
                                skipped_usecases.append(use_case)
                                continue
                                
                            print(f"Creating use case node from string: {use_case}")
                            result = session.run("""
                                MERGE (u:UseCase {name: $name})
                                ON CREATE SET u.created_at = datetime()
                                ON MATCH SET u.updated_at = datetime()
                                RETURN u.name
                            """, name=use_case)
                            
                            # Check if node was created
                            if result.peek():
                                stored_usecases.append(use_case)
                                print(f"Successfully created use case: {use_case}")
                            else:
                                print(f"WARNING: Failed to create use case: {use_case}")
                                skipped_usecases.append(use_case)
                        else:
                            print(f"WARNING: Invalid use case format: {type(use_case)} - {use_case}")
                            skipped_usecases.append(use_case)
                    except Exception as e:
                        print(f"ERROR processing use case #{i+1}: {str(e)}")
                        print(f"Problem use case: {use_case}")
                        skipped_usecases.append(use_case)
                
                print(f"Stored {len(stored_usecases)}/{len(analysis_result.get('use_cases', []))} use cases in Neo4j")
                print(f"Stored use cases: {stored_usecases}")
                if skipped_usecases:
                    print(f"Skipped {len(skipped_usecases)} use cases: {skipped_usecases}")

                # Compare with what's actually in the database
                db_usecases_result = session.run("MATCH (u:UseCase) RETURN u.name AS name ORDER BY u.name").data()
                db_usecases = [record['name'] for record in db_usecases_result]
                print(f"Use cases found in database: {len(db_usecases)}")
                print(f"Database use cases: {db_usecases}")
                
                # Check what's missing
                expected_names = set()
                for uc in analysis_result.get('use_cases', []):
                    if isinstance(uc, dict) and uc.get('name'):
                        expected_names.add(uc.get('name'))
                    elif isinstance(uc, str):
                        expected_names.add(uc)
                
                db_names = set(db_usecases)
                missing_names = expected_names - db_names
                if missing_names:
                    print(f"Missing use cases in database: {missing_names}")

                # First, verify which nodes exist before creating relationships
                # This helps debug issues with node matching
                existing_nodes_query = """
                    MATCH (n) 
                    WHERE n:Actor OR n:UseCase 
                    RETURN n.name AS name, labels(n) AS types
                """
                existing_nodes_result = session.run(existing_nodes_query).data()
                existing_node_names = {record['name']: record['types'] for record in existing_nodes_result}
                print(f"Found {len(existing_node_names)} existing nodes in Neo4j: {existing_node_names}")

                # Store Relationships
                successful_rels = 0
                total_rels = len(analysis_result.get('relationships', []))
                
                for rel in analysis_result.get('relationships', []):
                    # Support both from/to and source/target key formats
                    if isinstance(rel, dict) and rel.get('type'):
                        # Get source node (from or source)
                        from_node = None
                        if 'from' in rel:
                            from_node = rel['from']
                        elif 'source' in rel:
                            from_node = rel['source']
                            
                        # Get target node (to or target)
                        to_node = None
                        if 'to' in rel:
                            to_node = rel['to']
                        elif 'target' in rel:
                            to_node = rel['target']
                            
                        # Skip if either node is missing
                        if not from_node or not to_node:
                            print(f"WARNING: Skipping relationship with missing source/target: {rel}")
                            continue
                    else:
                        print(f"WARNING: Skipping invalid relationship data: {rel}")
                        continue

                    # Check if nodes for this relationship exist
                    if from_node not in existing_node_names:
                        print(f"WARNING: Source node '{from_node}' doesn't exist for relationship {rel}")
                        # Try to create it as actor (best guess)
                        session.run("""
                            MERGE (a:Actor {name: $name})
                            ON CREATE SET a.created_at = datetime(), a.auto_created = true
                        """, name=from_node)
                        existing_node_names[from_node] = ['Actor']
                    
                    if to_node not in existing_node_names:
                        print(f"WARNING: Target node '{to_node}' doesn't exist for relationship {rel}")
                        # Try to create it as use case (best guess)
                        create_query = """
                            MERGE (u:UseCase {name: $name})
                            ON CREATE SET u.created_at = datetime(), u.auto_created = true
                        """
                        session.run(create_query, name=to_node)
                        existing_node_names[to_node] = ['UseCase']

                    # Sanitize relationship type for Cypher (alphanumeric and underscore)
                    rel_type = rel['type'].lower()
                    rel_type_sanitized = re.sub(r'[^a-zA-Z0-9_]', '', rel_type.upper().replace(" ", "_"))
                    if not rel_type_sanitized:
                        rel_type_sanitized = "RELATED_TO" # Default if type becomes empty after sanitization
                        
                    # Prepare properties, removing source/target or from/to keys
                    rel_props = {k: v for k, v in rel.items() if k not in ['from', 'to', 'source', 'target', 'type']}
                    rel_props['original_type'] = rel['type'] # Store the original type if needed

                    # Improve relationship creation query
                    # Using CASE INSENSITIVE matching to handle case mismatches
                    query = f"""
                    MATCH (from_node)
                    WHERE toLower(from_node.name) = toLower($from_name) AND (from_node:Actor OR from_node:UseCase)
                    MATCH (to_node)
                    WHERE toLower(to_node.name) = toLower($to_name) AND (to_node:Actor OR to_node:UseCase)
                    MERGE (from_node)-[r:{rel_type_sanitized}]->(to_node)
                    ON CREATE SET r = $props, r.created_at = datetime()
                    ON MATCH SET r += $props, r.updated_at = datetime()
                    RETURN from_node.name, type(r), to_node.name
                    """
                    result = session.run(query, 
                                from_name=from_node, 
                                to_name=to_node, 
                                props=rel_props)
                    
                    # Check if relationship was created successfully
                    if result.peek():
                        successful_rels += 1
                        record = result.single()
                        print(f"Created relationship: {record[0]} --[{record[1]}]--> {record[2]}")
                
                print(f"Successfully stored {successful_rels}/{total_rels} relationships in Neo4j.")
                
                # Query all relationships for verification
                all_relationships = session.run("""
                    MATCH (src)-[r]->(dst)
                    RETURN src.name AS source, type(r) AS type, dst.name AS target
                """).data()
                
                print(f"Total relationships in database: {len(all_relationships)}")
                for i, rel in enumerate(all_relationships):
                    print(f"  Relationship {i+1}: {rel['source']} --[{rel['type']}]--> {rel['target']}")
                
                return True
        except Exception as e:
            # Adding more specific error logging for Neo4j issues.
            print(f"ERROR: Error storing analysis results in Neo4j: {str(e)}")
            import traceback
            print(traceback.format_exc()) # Log full traceback for Neo4j errors
            return False

    def create_usecase_node(self, usecase: Dict[str, Any]) -> None:
        """Create a use case node in Neo4j."""
        try:
            query = """
            CREATE (u:UseCase {
                name: $name,
                description: $description,
                type: $type,
                preconditions: $preconditions,
                postconditions: $postconditions,
                priority: $priority
            })
            """
            
            params = {
                "name": usecase["name"],
                "description": usecase.get("description", ""),
                "type": usecase.get("type", "main"),
                "preconditions": usecase.get("preconditions", ["No preconditions specified"]),
                "postconditions": usecase.get("postconditions", ["No postconditions specified"]),
                "priority": usecase.get("priority", "medium")
            }
            
            self.session.run(query, params)
            logger.info(f"Created use case node: {usecase['name']}")
            
        except Exception as e:
            logger.error(f"Error creating use case node: {str(e)}")
            raise

    def export_graph(self, project_name: str):
        """Export the entire graph to a JSON file in the project directory"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n, r, m
                """)
                
                graph_data = {
                    "nodes": [],
                    "relationships": []
                }
                
                processed_nodes = set()

                for record in result:
                    nodes_to_process = []
                    if record["n"]:
                        nodes_to_process.append(record["n"])
                    if record["m"]:
                        nodes_to_process.append(record["m"])

                    for node_obj in nodes_to_process:
                        if node_obj and node_obj.id not in processed_nodes:
                            node_props = dict(node_obj)
                            for key, value in node_props.items():
                                if isinstance(value, DateTime):
                                    node_props[key] = value.isoformat()
                                elif isinstance(value, list):
                                    # Handle lists, converting DateTime if they appear in lists
                                    node_props[key] = [
                                        item.isoformat() if isinstance(item, DateTime) else item 
                                        for item in value
                                    ]
                                    
                            graph_data["nodes"].append({
                                "id": node_obj.id,
                                "labels": list(node_obj.labels),
                                "properties": node_props
                            })
                            processed_nodes.add(node_obj.id)
                    
                    if record["r"]:
                        rel_props = dict(record["r"])
                        for key, value in rel_props.items():
                            if isinstance(value, DateTime):
                                rel_props[key] = value.isoformat()
                            elif isinstance(value, list):
                                rel_props[key] = [
                                    item.isoformat() if isinstance(item, DateTime) else item 
                                    for item in value
                                ]

                        graph_data["relationships"].append({
                            "id": record["r"].id, # Adding relationship ID
                            "start_node_id": record["r"].start_node.id,
                            "end_node_id": record["r"].end_node.id,
                            "type": record["r"].type,
                            "properties": rel_props
                        })
                
                # Ensure output directory is using the global OUTPUT_DIR from main.py if available
                # For robustness, if this module is used standalone, default to a known path.
                # This assumes main.py sets a global OUTPUT_DIR. If not, this needs adjustment.
                # A better way would be to pass OUTPUT_DIR to this function.
                # For now, deriving from current working directory structure if possible.
                base_output_dir = os.getenv("OUTPUT_DIR", "output") # Fallback
                # Try to get the numbered output_dir if this is run via main.py
                # This is a bit of a hack; direct passing of output_dir to export_graph would be cleaner.
                current_script_path = os.path.dirname(os.path.abspath(__file__))
                main_output_dir_segment = os.path.basename(os.getcwd()) # e.g. 'UMLify' or a numbered folder
                
                # This logic for project_dir needs to be robust or receive the correct path
                # from the caller (main.py)
                # Using a simplified approach assuming it runs in context of main.py's OUTPUT_DIR
                # The `project_name` argument can be used to create a subfolder within the neo4j export dir

                # Determine the base Neo4j export directory from OUTPUT_DIR in main.py context
                # This assumes that the CWD or an env var correctly points to the numbered output folder
                # or this function is called by main.py which has already set up OUTPUT_DIR.

                # Get the main script's output directory. This is tricky if Neo4jOperations is used independently.
                # A more direct way: OUTPUT_DIR should be passed from main.py to PlantUMLGenerator and then to Neo4jOperations if needed for export.
                # For now, we assume OUTPUT_DIR is accessible or we use a default.
                if 'output' in base_output_dir: # if OUTPUT_DIR is like "output/1"
                    neo4j_export_base_dir = os.path.join(base_output_dir, "neo4j")
                else: # Fallback if OUTPUT_DIR is not set as expected
                    neo4j_export_base_dir = os.path.join("output", "neo4j_exports")

                project_export_dir = os.path.join(neo4j_export_base_dir, project_name.lower().replace(' ', '_'))
                os.makedirs(project_export_dir, exist_ok=True)
                
                output_path = os.path.join(project_export_dir, "graph_export.json")
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Graph exported to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting graph: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_relationships(self, query_params=None):
        """
        Get all relationships from the Neo4j database.
        
        Args:
            query_params (dict, optional): Optional query parameters
            
        Returns:
            list: List of relationships
        """
        try:
            query = """
            MATCH (source)-[r]->(target)
            RETURN source.name AS source, target.name AS target, type(r) AS type
            """
            
            if query_params:
                # Add parameters to query if provided
                conditions = []
                params = {}
                
                if "source" in query_params:
                    conditions.append("source.name = $source_name")
                    params["source_name"] = query_params["source"]
                    
                if "target" in query_params:
                    conditions.append("target.name = $target_name")
                    params["target_name"] = query_params["target"]
                    
                if "type" in query_params:
                    conditions.append("type(r) = $rel_type")
                    params["rel_type"] = query_params["type"]
                    
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                relationships = self.driver.session().run(query, params).data()
            else:
                relationships = self.driver.session().run(query).data()
                
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []

    def get_specific_relationships(self, relationship_type):
        """
        Get specific type of relationships from the Neo4j database.
        
        Args:
            relationship_type (str): Type of relationship to retrieve (e.g., "include", "extend", "generalization")
            
        Returns:
            list: List of relationships of the specified type
        """
        try:
            query = """
            MATCH (source)-[r]->(target)
            WHERE type(r) = $rel_type OR type(r) = $rel_type_upper
            RETURN source.name AS source, target.name AS target, type(r) AS type
            """
            
            # Account for different casing conventions
            params = {
                "rel_type": relationship_type.lower(),
                "rel_type_upper": relationship_type.upper()
            }
            
            relationships = self.driver.session().run(query, params).data()
            logger.debug(f"Retrieved {len(relationships)} {relationship_type} relationships")
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting {relationship_type} relationships: {str(e)}")
            return []

    def get_patterns(self):
        """
        Get UML patterns from the Neo4j database.
        
        Returns:
            list: List of UML patterns
        """
        try:
            query = """
            MATCH (p:Pattern)
            RETURN p.description AS pattern
            """
            
            result = self.driver.session().run(query).data()
            patterns = [item["pattern"] for item in result]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting patterns: {str(e)}")
            return []

    def store_validation_rule(self, rule_data: Dict[str, Any]) -> str:
        """Store a validation rule in the knowledge graph."""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (r:ValidationRule {name: $name})
                ON CREATE SET r.description = $description,
                             r.conditions = $conditions,
                             r.created_at = datetime()
                ON MATCH SET r.description = $description,
                            r.conditions = $conditions,
                            r.updated_at = datetime()
                RETURN r.name
                """,
                rule_data
            )
            return result.single()[0]

    def store_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Store a UML pattern in the knowledge graph."""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (p:Pattern {name: $name})
                ON CREATE SET p.description = $description,
                             p.validation_rules = $validation_rules,
                             p.created_at = datetime()
                ON MATCH SET p.description = $description,
                            p.validation_rules = $validation_rules,
                            p.updated_at = datetime()
                RETURN p.name
                """,
                pattern_data
            )
            return result.single()[0]

    def validate_use_case(self, usecase_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a use case against stored patterns and rules."""
        with self.driver.session() as session:
            # Get validation rules
            rules = session.run(
                """
                MATCH (r:ValidationRule)
                RETURN r.name as name, r.conditions as conditions
                """
            ).data()
            
            # Get patterns
            patterns = session.run(
                """
                MATCH (p:Pattern)
                RETURN p.name as name, p.validation_rules as validation_rules
                """
            ).data()
            
            # Perform validation
            validation_results = {
                "is_valid": True,
                "violations": [],
                "suggestions": []
            }
            
            # Check against rules
            for rule in rules:
                conditions = rule.get("conditions", [])
                for condition in conditions:
                    if not self._check_condition(usecase_data, condition):
                        validation_results["is_valid"] = False
                        validation_results["violations"].append({
                            "rule": rule["name"],
                            "condition": condition
                        })
            
            # Check against patterns
            for pattern in patterns:
                pattern_rules = pattern.get("validation_rules", [])
                for rule in pattern_rules:
                    if not self._check_pattern_rule(usecase_data, rule):
                        validation_results["suggestions"].append({
                            "pattern": pattern["name"],
                            "suggestion": rule
                        })
            
            return validation_results

    def _check_condition(self, data: Dict[str, Any], condition: str) -> bool:
        """Check if data satisfies a condition."""
        try:
            # Simple condition checking - can be enhanced with more complex logic
            if "required" in condition:
                field = condition["required"]
                return field in data and data[field] is not None
            elif "pattern" in condition:
                field = condition["field"]
                pattern = condition["pattern"]
                return field in data and re.match(pattern, str(data[field]))
            return True
        except Exception:
            return False

    def _check_pattern_rule(self, data: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if data follows a pattern rule."""
        try:
            # Pattern rule checking - can be enhanced with more complex logic
            if "required_fields" in rule:
                return all(field in data for field in rule["required_fields"])
            elif "field_patterns" in rule:
                for field, pattern in rule["field_patterns"].items():
                    if field not in data or not re.match(pattern, str(data[field])):
                        return False
            return True
        except Exception:
            return False 