import networkx as nx
from scipy.spatial.distance import cosine

class PhilosophicalPatternDetector:
    """
    A neural constellation finder that maps the hidden connections 
    between philosophical ideas! 
    """
    def __init__(self, similarity_threshold=0.7):
        self.concept_graph = nx.DiGraph()
        self.similarity_threshold = similarity_threshold
        self.pattern_cache = {}
        
    def detect_emergent_patterns(
        self,
        concept_tracker: ConceptTracker,
        current_round: str
    ) -> List[Dict[str, Any]]:
        """
        Discovers emergent philosophical patterns like a cognitive astronomer 
        finding new constellations in the night sky of ideas!
        """
        # Build concept similarity network
        concepts = list(concept_tracker.concept_embeddings.keys())
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                similarity = self._calculate_concept_similarity(
                    concept_tracker.concept_embeddings[concept1],
                    concept_tracker.concept_embeddings[concept2]
                )
                
                if similarity > self.similarity_threshold:
                    self.concept_graph.add_edge(
                        concept1, 
                        concept2, 
                        weight=similarity
                    )
        
        # Find emergent patterns using graph analysis
        patterns = []
        
        # Look for highly connected components (concept clusters)
        for component in nx.connected_components(self.concept_graph.to_undirected()):
            if len(component) >= 3:  # Minimum size for a pattern
                subgraph = self.concept_graph.subgraph(component)
                
                # Calculate pattern strength using centrality
                centrality_scores = nx.pagerank(subgraph)
                pattern_strength = np.mean(list(centrality_scores.values()))
                
                pattern = {
                    'concepts': list(component),
                    'strength': pattern_strength,
                    'central_concept': max(centrality_scores.items(), key=lambda x: x[1])[0]
                }
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x['strength'], reverse=True)
