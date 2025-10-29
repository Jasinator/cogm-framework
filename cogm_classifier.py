"""
COGM Classifier: Prompt-to-Quadrant Classification
===================================================

Implements the COGM (Cognitive Optimization for Generative Models) framework
for classifying LLM prompts into quadrants based on:
- X-axis: TPN (analytical) vs DMN (reflective) processing
- Y-axis: Outcome-focused vs Process-focused orientation

Based on paper: "Task-Adaptive Resource Allocation for LLMs:
Heterogeneous Routing via a Cognitive Task Ontology"

Author: Jason Ader
License: MIT
"""

import numpy as np
from typing import Tuple, Dict
import warnings

# Optional: suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class COGMClassifier:
    """
    COGM quadrant classifier using embedding-based projection.
    
    Note: This is a conceptual implementation. Axis vectors (tpn_dmn_axis, 
    outcome_process_axis) should be learned from labeled training data in 
    production use. Current implementation uses placeholder random vectors.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize COGM classifier.
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = 384  # for all-MiniLM-L6-v2
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        # Initialize axis vectors (PLACEHOLDER - should be learned from data)
        # In production: Load pre-trained axis vectors from file
        np.random.seed(42)  # For reproducibility in demo
        self.tpn_dmn_axis = self._normalize(np.random.randn(self.embedding_dim))
        self.outcome_process_axis = self._normalize(np.random.randn(self.embedding_dim))
        
        # Quadrant definitions from Table 1
        self.quadrant_names = {
            (-1, 1): 'Analytical Execution',    # TPN + Outcome
            (-1, -1): 'Structured Refinement',  # TPN + Process
            (1, 1): 'Narrative Engagement',     # DMN + Outcome
            (1, -1): 'Creative Synthesis'       # DMN + Process
        }
        
        # Base resource allocations (GPU%, SNN%)
        self.base_splits = {
            'Analytical Execution': (0.90, 0.10),
            'Structured Refinement': (0.80, 0.20),
            'Narrative Engagement': (0.20, 0.80),
            'Creative Synthesis': (0.30, 0.70)
        }
    
    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        return vector / np.linalg.norm(vector)
    
    def classify_prompt(self, prompt_text: str) -> Tuple[str, float, float, float]:
        """
        Classify prompt into COGM quadrant via embedding projection.
        
        Args:
            prompt_text: User input prompt to classify
            
        Returns:
            Tuple of (quadrant_name, x_score, y_score, confidence)
            - quadrant_name: One of 4 COGM quadrants
            - x_score: TPN-DMN axis position in [-1, 1] (negative=TPN, positive=DMN)
            - y_score: Outcome-Process position in [-1, 1] (negative=Process, positive=Outcome)
            - confidence: Classification confidence in [0, 1]
        """
        # Generate and normalize prompt embedding
        embedding = self.model.encode(prompt_text)
        embedding = self._normalize(embedding)
        
        # Project onto cognitive axes
        x_score = np.dot(embedding, self.tpn_dmn_axis)
        y_score = np.dot(embedding, self.outcome_process_axis)
        
        # Determine quadrant (hard classification)
        x_sign = -1 if x_score < 0 else 1
        y_sign = 1 if y_score > 0 else -1
        quadrant = self.quadrant_names[(x_sign, y_sign)]
        
        # Compute classification confidence (distance from origin)
        confidence = np.sqrt(x_score**2 + y_score**2)
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        return quadrant, x_score, y_score, confidence
    
    def get_resource_split(self, quadrant: str, x_score: float = None, 
                          y_score: float = None) -> Tuple[float, float]:
        """
        Get GPU/SNN resource allocation for a given quadrant.
        
        Args:
            quadrant: COGM quadrant name
            x_score: Optional TPN-DMN score for soft boundaries
            y_score: Optional Outcome-Process score for soft boundaries
            
        Returns:
            Tuple of (gpu_percentage, snn_percentage) in [0, 1]
        """
        gpu_base, snn_base = self.base_splits[quadrant]
        
        # Optional: Implement soft boundaries using x_score/y_score
        # For now, using hard quadrant boundaries
        
        return gpu_base, snn_base
    
    def classify_and_route(self, prompt_text: str) -> Dict[str, any]:
        """
        Full pipeline: classify prompt and determine resource allocation.
        
        Args:
            prompt_text: Input prompt to classify and route
            
        Returns:
            Dictionary with classification results and routing decision
        """
        quadrant, x, y, conf = self.classify_prompt(prompt_text)
        gpu_pct, snn_pct = self.get_resource_split(quadrant, x, y)
        
        return {
            'prompt': prompt_text,
            'quadrant': quadrant,
            'x_score': x,
            'y_score': y,
            'confidence': conf,
            'gpu_allocation': gpu_pct,
            'snn_allocation': snn_pct,
            'axis_interpretation': {
                'x_axis': 'TPN (Analytical)' if x < 0 else 'DMN (Reflective)',
                'y_axis': 'Outcome-focused' if y > 0 else 'Process-focused'
            }
        }


def train_axis_vectors(prompts: list, labels: dict, embedding_dim: int = 384) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train COGM axis vectors from labeled data (PLACEHOLDER STUB).
    
    In production, this would:
    1. Collect diverse prompts (N > 10,000) with expert labels
    2. Generate embeddings for all prompts
    3. Learn axis vectors via linear regression or contrastive learning
    4. Validate on held-out test set
    
    Args:
        prompts: List of prompt strings
        labels: Dict with 'tpn_dmn' and 'outcome_process' scores for each prompt
        embedding_dim: Dimensionality of embeddings
        
    Returns:
        Tuple of (tpn_dmn_axis, outcome_process_axis) as normalized vectors
    """
    # STUB: In production, implement supervised learning here
    # For now, return random vectors
    np.random.seed(42)
    tpn_dmn_axis = np.random.randn(embedding_dim)
    outcome_process_axis = np.random.randn(embedding_dim)
    
    # Normalize
    tpn_dmn_axis = tpn_dmn_axis / np.linalg.norm(tpn_dmn_axis)
    outcome_process_axis = outcome_process_axis / np.linalg.norm(outcome_process_axis)
    
    return tpn_dmn_axis, outcome_process_axis


# Example usage and testing
def main():
    """Demonstrate COGM classifier with example prompts."""
    
    print("="*70)
    print("COGM Classifier Demo")
    print("="*70)
    print("\nInitializing classifier...")
    
    classifier = COGMClassifier()
    
    # Test prompts representing different quadrants
    test_prompts = [
        ("Analyze the quarterly financial report and extract key metrics.", "Analytical Execution"),
        ("Debug this Python code and explain each error.", "Analytical Execution"),
        ("Optimize the database query for better performance.", "Structured Refinement"),
        ("Write a persuasive essay about climate change.", "Narrative Engagement"),
        ("Brainstorm creative names for a tech startup.", "Creative Synthesis"),
        ("Generate story ideas exploring themes of identity.", "Creative Synthesis"),
    ]
    
    print(f"\nClassifying {len(test_prompts)} example prompts...\n")
    
    for i, (prompt, expected) in enumerate(test_prompts, 1):
        print(f"Example {i}:")
        print(f"Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        
        result = classifier.classify_and_route(prompt)
        
        print(f"  Quadrant: {result['quadrant']}")
        print(f"  Position: X={result['x_score']:.3f} ({result['axis_interpretation']['x_axis']})")
        print(f"            Y={result['y_score']:.3f} ({result['axis_interpretation']['y_axis']})")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Allocation: {result['gpu_allocation']*100:.0f}% GPU, {result['snn_allocation']*100:.0f}% SNN")
        print(f"  Expected: {expected}")
        print()
    
    print("="*70)
    print("NOTE: This demo uses placeholder axis vectors.")
    print("In production, axis vectors should be learned from labeled training data.")
    print("="*70)


if __name__ == "__main__":
    main()
