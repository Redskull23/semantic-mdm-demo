"""
Semantic MDM - Vector Search Module
Handles semantic similarity search across domain terminology
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class SemanticMDM:
    """
    Semantic Master Data Management using vector embeddings
    for cross-domain terminology alignment
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Semantic MDM system
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       'all-MiniLM-L6-v2' is fast and good quality (default)
                       'all-mpnet-base-v2' is slower but higher quality
        """
        print(f"🔧 Loading semantic model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✅ Model loaded successfully")
        
        # Domain term storage
        self.legal_terms: List[Dict[str, Any]] = []
        self.restdev_terms: List[Dict[str, Any]] = []
        self.finance_terms: List[Dict[str, Any]] = []
        self.all_terms: List[Dict[str, Any]] = []
        
        # Vector embeddings
        self.term_embeddings: Optional[np.ndarray] = None
        
        # Statistics
        self.stats = {
            'total_terms': 0,
            'legal_terms': 0,
            'restdev_terms': 0,
            'finance_terms': 0
        }
    
    def load_domain_data(self) -> None:
        """Load domain terminology from JSON files"""
        data_dir = Path(__file__).parent.parent / "data"
        
        print(f"\n📁 Loading domain data from: {data_dir}")
        
        # Load Legal terms
        self.legal_terms = self._load_json_terms(
            data_dir / "legal_terms.json",
            domain_name="Legal"
        )
        
        # Load Restaurant Development terms
        self.restdev_terms = self._load_json_terms(
            data_dir / "restdev_terms.json",
            domain_name="Restaurant Development"
        )
        
        # Load Finance terms (optional)
        self.finance_terms = self._load_json_terms(
            data_dir / "finance_terms.json",
            domain_name="Finance",
            required=False
        )
        
        # Combine all terms
        self.all_terms = self.legal_terms + self.restdev_terms + self.finance_terms
        
        # Update statistics
        self.stats = {
            'total_terms': len(self.all_terms),
            'legal_terms': len(self.legal_terms),
            'restdev_terms': len(self.restdev_terms),
            'finance_terms': len(self.finance_terms)
        }
        
        # Create embeddings
        if self.all_terms:
            self._create_embeddings()
            self._print_loading_summary()
        else:
            print("❌ Error: No terms loaded!")
    
    def _load_json_terms(
        self, 
        filepath: Path, 
        domain_name: str,
        required: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load terms from a JSON file
        
        Args:
            filepath: Path to JSON file
            domain_name: Name of the domain (e.g., "Legal")
            required: Whether this file is required
            
        Returns:
            List of term dictionaries with domain field added
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            # Handle both formats:
            # 1. {"domain": "Legal", "terms": [...]}
            # 2. [{"term": "...", "domain": "Legal"}, ...]
            
            if isinstance(data, dict) and "terms" in data:
                # Format 1: Nested structure
                terms = []
                for term_data in data["terms"]:
                    # Add domain field if not present
                    if "domain" not in term_data:
                        term_data["domain"] = data.get("domain", domain_name)
                    terms.append(term_data)
                print(f"  ✅ Loaded {len(terms)} terms from {filepath.name}")
                return terms
                
            elif isinstance(data, list):
                # Format 2: Flat list
                # Ensure each term has domain field
                for term_data in data:
                    if "domain" not in term_data:
                        term_data["domain"] = domain_name
                print(f"  ✅ Loaded {len(data)} terms from {filepath.name}")
                return data
                
            else:
                print(f"  ⚠️  Invalid format in {filepath.name}")
                return []
                
        except FileNotFoundError:
            if required:
                print(f"  ❌ Required file not found: {filepath.name}")
            else:
                print(f"  ℹ️  Optional file not found: {filepath.name}")
            return []
            
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON decode error in {filepath.name}: {e}")
            return []
            
        except Exception as e:
            print(f"  ❌ Error loading {filepath.name}: {e}")
            return []
    
    def _create_embeddings(self) -> None:
        """Create vector embeddings for all terms"""
        print(f"\n🧮 Creating embeddings for {len(self.all_terms)} terms...")
        
        # Extract term text
        terms_text = [item['term'] for item in self.all_terms]
        
        # Create embeddings
        self.term_embeddings = self.model.encode(
            terms_text,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Created embeddings: {self.term_embeddings.shape}")
    
    def _print_loading_summary(self) -> None:
        """Print summary of loaded data"""
        print("\n" + "="*60)
        print("📊 LOADING SUMMARY")
        print("="*60)
        print(f"  Legal Terms:       {self.stats['legal_terms']:>4}")
        print(f"  RestDev Terms:     {self.stats['restdev_terms']:>4}")
        print(f"  Finance Terms:     {self.stats['finance_terms']:>4}")
        print(f"  {'─'*56}")
        print(f"  Total Terms:       {self.stats['total_terms']:>4}")
        print("="*60 + "\n")
    
    def semantic_search(
        self,
        query: str,
        domain: str = "All Domains",
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across domain terminology
        
        Args:
            query: Search string
            domain: Filter by domain (e.g., "Legal", "Restaurant Development")
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
        Returns:
            List of matching terms with similarity scores
        """
        if not self.all_terms or self.term_embeddings is None:
            print("❌ No terms loaded. Call load_domain_data() first.")
            return []
        
        # Create embedding for query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.term_embeddings)[0]
        
        # Create results with all term data
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                term_data = self.all_terms[idx].copy()
                term_data['similarity'] = float(similarity)
                results.append(term_data)
        
        # Filter by domain if specified
        if domain != "All Domains":
            results = [r for r in results if r['domain'] == domain]
        
        # Sort by similarity (highest first)
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def get_canonical_mapping(self, term: str) -> Optional[str]:
        """
        Map a term to its canonical taxonomy concept
        
        Args:
            term: Term to map
            
        Returns:
            Canonical concept path or None
        """
        results = self.semantic_search(term, top_k=1)
        if results:
            return results[0].get('canonical')
        return None
    
    def find_cross_domain_matches(
        self,
        term: str,
        source_domain: str,
        min_similarity: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find semantically similar terms in other domains
        
        Args:
            term: Source term to match
            source_domain: Domain of the source term
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary of matches grouped by domain
        """
        # Get all similar terms
        all_matches = self.semantic_search(
            query=term,
            domain="All Domains",
            top_k=50,
            min_similarity=min_similarity
        )
        
        # Group by domain, excluding source domain
        matches_by_domain = {}
        for match in all_matches:
            domain = match['domain']
            if domain != source_domain:
                if domain not in matches_by_domain:
                    matches_by_domain[domain] = []
                matches_by_domain[domain].append(match)
        
        return matches_by_domain
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded data"""
        return self.stats.copy()
    
    def get_all_canonical_concepts(self) -> List[str]:
        """Get unique list of all canonical concepts"""
        concepts = set()
        for term in self.all_terms:
            if 'canonical' in term:
                concepts.add(term['canonical'])
        return sorted(list(concepts))
    
    def search_by_canonical(self, canonical_path: str) -> List[Dict[str, Any]]:
        """
        Find all terms that map to a specific canonical concept
        
        Args:
            canonical_path: Canonical taxonomy path (e.g., "Agreements > Real Estate > Ground Lease")
            
        Returns:
            List of terms that map to this concept
        """
        matches = []
        for term in self.all_terms:
            if term.get('canonical') == canonical_path:
                matches.append(term.copy())
        return matches


# Example usage and testing
if __name__ == "__main__":
    # Test the semantic MDM
    print("🧪 Testing Semantic MDM...")
    
    mdm = SemanticMDM()
    mdm.load_domain_data()
    
    # Test search
    print("\n🔍 Testing search for 'ground lease':")
    results = mdm.semantic_search("ground lease", top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['term']}")
        print(f"   Domain: {result['domain']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Canonical: {result['canonical']}")
    
    # Test cross-domain matching
    print("\n\n🔄 Testing cross-domain matches for 'ground lease' from Legal:")
    matches = mdm.find_cross_domain_matches("ground lease", "Legal", min_similarity=0.7)
    
    for domain, terms in matches.items():
        print(f"\n{domain}:")
        for term in terms[:3]:  # Show top 3
            print(f"  - {term['term']} (similarity: {term['similarity']:.3f})")
    
    # Test canonical mapping
    print("\n\n📍 Testing canonical mapping:")
    canonical = mdm.get_canonical_mapping("site lease")
    print(f"'site lease' maps to: {canonical}")
    
    print("\n✅ Testing complete!")