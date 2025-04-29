from neo4j import GraphDatabase
import logging
from typing import Dict, Any, List
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        logger.info(f"Attempting to connect to Neo4j at {uri}")
        logger.info(f"Using username: {user}")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test the connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                logger.info("Successfully connected to Neo4j")
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
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Functionality) REQUIRE f.name IS UNIQUE")
                logger.info("Successfully set up Neo4j constraints")
        except Exception as e:
            logger.error(f"Failed to set up constraints: {str(e)}")
            raise
            
    def store_actor(self, actor: Dict[str, Any]):
        """Store an actor in the graph."""
        with self.driver.session() as session:
            session.run("""
                MERGE (a:Actor {name: $name})
                SET a.type = $type,
                    a.description = $description
            """, actor)
            
    def store_use_case(self, use_case: Dict[str, Any]):
        """Store a use case and its relationships."""
        with self.driver.session() as session:
            # Create use case
            session.run("""
                MERGE (u:UseCase {name: $name})
                SET u.description = $description,
                    u.preconditions = $preconditions,
                    u.postconditions = $postconditions
            """, use_case)
            
            # Create relationships with actors
            for actor_name in use_case.get('actors', []):
                session.run("""
                    MATCH (a:Actor {name: $actor_name})
                    MATCH (u:UseCase {name: $use_case_name})
                    MERGE (a)-[:PARTICIPATES_IN]->(u)
                """, {
                    'actor_name': actor_name,
                    'use_case_name': use_case['name']
                })
                
    def store_requirement(self, requirement: Dict[str, Any]):
        """Store a requirement."""
        # Generate a unique ID using Python's uuid
        requirement_id = str(uuid.uuid4())
        requirement['id'] = requirement_id
        
        with self.driver.session() as session:
            session.run("""
                CREATE (r:Requirement {
                    id: $id,
                    type: $type,
                    description: $description
                })
            """, requirement)
            
    def store_functionality(self, functionality: Dict[str, Any]):
        """Store a core functionality."""
        with self.driver.session() as session:
            session.run("""
                MERGE (f:Functionality {name: $name})
                SET f.description = $description
            """, functionality)
            
    def get_graph_data(self) -> Dict[str, Any]:
        """Get all graph data for UML generation."""
        try:
            with self.driver.session() as session:
                # Get actors
                actors = session.run("""
                    MATCH (a:Actor)
                    RETURN a.name as name, a.type as type, a.description as description
                """).data()
                
                # Get use cases
                use_cases = session.run("""
                    MATCH (u:UseCase)
                    RETURN u.name as name, u.description as description,
                           u.preconditions as preconditions, u.postconditions as postconditions
                """).data()
                
                # Get actor-use case relationships
                relationships = session.run("""
                    MATCH (a:Actor)-[r:PARTICIPATES_IN]->(u:UseCase)
                    RETURN a.name as actor, u.name as use_case, type(r) as relationship
                """).data()
                
                # Get requirements
                requirements = session.run("""
                    MATCH (r:Requirement)
                    RETURN r.type as type, r.description as description
                """).data()
                
                # Get functionalities
                functionalities = session.run("""
                    MATCH (f:Functionality)
                    RETURN f.name as name, f.description as description
                """).data()
                
                logger.info(f"Retrieved {len(actors)} actors, {len(use_cases)} use cases, {len(relationships)} relationships, {len(requirements)} requirements, and {len(functionalities)} functionalities")
                return {
                    "actors": actors,
                    "use_cases": use_cases,
                    "relationships": relationships,
                    "requirements": requirements,
                    "functionalities": functionalities
                }
        except Exception as e:
            logger.error(f"Failed to get graph data: {str(e)}")
            raise
            
    def close(self):
        """Close the Neo4j connection."""
        try:
            self.driver.close()
            logger.info("Successfully closed Neo4j connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
            raise 