import os
import json
import logging
import argparse
from dotenv import load_dotenv
from neo4j_operations import Neo4jOperations
from validation.agents import UMLEvaluator
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_uml_elements(filepath: str):
    """Load UML elements from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading UML elements from {filepath}: {str(e)}")
        return None

def plot_comparison_results(results, output_path=None):
    """Plot comparison results as a radar chart."""
    metrics = [
        'ged_score', 'bert_score', 
        'actor_precision', 'actor_recall', 
        'usecase_precision', 'usecase_recall'
    ]
    
    values = [results[m] for m in metrics]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Close the loop
    angles += angles[:1]  # Close the loop
    
    labels = [m.replace('_', ' ').title() for m in metrics]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    # Set chart title
    ax.set_title("UML Diagram Comparison Metrics", size=15, pad=20)
    
    # Set y-axis limit
    ax.set_ylim(0, 1)
    
    # Add a grid
    ax.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved comparison chart to {output_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    """Compare two UML diagrams using evaluation metrics."""
    parser = argparse.ArgumentParser(description="Compare UML diagrams using evaluation metrics")
    parser.add_argument("--reference", "-r", required=True, help="Path to reference UML elements JSON")
    parser.add_argument("--generated", "-g", required=True, help="Path to generated UML elements JSON")
    parser.add_argument("--output", "-o", help="Path to save comparison results")
    parser.add_argument("--plot", "-p", help="Path to save comparison chart")
    args = parser.parse_args()
    
    # Load environment variables for Neo4j connection
    load_dotenv()
    
    # Get Neo4j credentials from environment
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        logger.error("Missing Neo4j credentials in environment variables")
        return
    
    # Load UML elements
    reference_elements = load_uml_elements(args.reference)
    generated_elements = load_uml_elements(args.generated)
    
    if not reference_elements or not generated_elements:
        logger.error("Failed to load UML elements")
        return
    
    # Initialize Neo4j connection
    neo4j_ops = None
    try:
        neo4j_ops = Neo4jOperations(neo4j_uri, neo4j_user, neo4j_password)
        
        # Initialize UML evaluator
        uml_evaluator = UMLEvaluator(neo4j_ops)
        
        # Compare diagrams
        comparison_results = uml_evaluator.compare_diagrams(
            reference_elements=reference_elements,
            generated_elements=generated_elements
        )
        
        # Print comparison results
        logger.info("Comparison results:")
        logger.info(f"- Graph Edit Distance (GED) score: {comparison_results['ged_score']:.4f}")
        logger.info(f"- Semantic similarity (BERTScore): {comparison_results['bert_score']:.4f}")
        logger.info(f"- Actor precision: {comparison_results['actor_precision']:.4f}")
        logger.info(f"- Actor recall: {comparison_results['actor_recall']:.4f}")
        logger.info(f"- Actor F1: {comparison_results['actor_f1']:.4f}")
        logger.info(f"- Use case precision: {comparison_results['usecase_precision']:.4f}")
        logger.info(f"- Use case recall: {comparison_results['usecase_recall']:.4f}")
        logger.info(f"- Use case F1: {comparison_results['usecase_f1']:.4f}")
        
        # Save comparison results if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved comparison results to {args.output}")
        
        # Plot comparison results if plot path provided
        if args.plot:
            plot_comparison_results(comparison_results, args.plot)
    
    except Exception as e:
        logger.error(f"Error comparing UML diagrams: {str(e)}")
    finally:
        # Close Neo4j connection if initialized
        if neo4j_ops is not None:
            neo4j_ops.close()

if __name__ == "__main__":
    main() 