from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
from datetime import datetime

class ConceptTracker:
    """
    A sophisticated system for tracking philosophical concept evolution,
    like a temporal telescope observing the birth and growth of ideas!
    """
    def __init__(self, embedding_dim=100):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)  # Capture compound concepts!
        )
        self.concept_embeddings = {}
        self.temporal_weights = {}
        self.concept_graph = nx.DiGraph()  # For tracking concept relationships
        
    def process_dialogue_round(
        self, 
        round_text: str, 
        timestamp: datetime,
        decay_factor: float = 0.95
    ) -> Dict[str, float]:
        """
        Processes a round of philosophical dialogue, updating our concept space
        like a cognitive cartographer mapping new territories of thought!
        """
        # Extract key concepts using TF-IDF
        tfidf_matrix = self.tfidf.fit_transform([round_text])
        feature_names = self.tfidf.get_feature_names_out()
        
        # Calculate temporal importance
        temporal_scores = {}
        current_time = datetime.now()
        
        for concept, old_weight in self.temporal_weights.items():
            # Apply temporal decay to existing concepts
            time_delta = (current_time - timestamp).total_seconds() / 3600  # hours
            decay = decay_factor ** time_delta
            self.temporal_weights[concept] = old_weight * decay
        
        # Update with new concepts
        for idx, score in enumerate(tfidf_matrix.toarray()[0]):
            concept = feature_names[idx]
            if score > 0:
                if concept not in self.temporal_weights:
                    self.temporal_weights[concept] = score
                else:
                    # Reinforcement learning-style update
                    self.temporal_weights[concept] += score * (1 - self.temporal_weights[concept])
                
                # Update concept embedding
                self._update_concept_embedding(concept, round_text)
        
        return self.temporal_weights

    def _update_concept_embedding(self, concept: str, context: str):
        """
        Updates our understanding of a concept based on its current usage context,
        like a philosophical GPS tracking idea evolution!
        """
        # Create or update concept embedding
        words = context.lower().split()
        try:
            model = Word2Vec([words], vector_size=100, window=5, min_count=1)
            if concept in model.wv:
                if concept not in self.concept_embeddings:
                    self.concept_embeddings[concept] = model.wv[concept]
                else:
                    # Exponential moving average update
                    alpha = 0.1
                    self.concept_embeddings[concept] = (
                        (1 - alpha) * self.concept_embeddings[concept] +
                        alpha * model.wv[concept]
                    )
        except Exception as e:
            print(f"Warning: Embedding update failed for concept '{concept}': {e}")
