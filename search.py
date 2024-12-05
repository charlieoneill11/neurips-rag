import numpy as np
from typing import List, Dict, Tuple
import pickle
import glob
from openai import OpenAI
import yaml
import tiktoken

class PosterSearch:
    def __init__(self, embeddings_dir: str = "embeddings", embedding_model: str = "text-embedding-3-small"):
        # Initialize OpenAI client
        config = yaml.safe_load(open('config.yaml', 'r'))
        self.client = OpenAI(api_key=config['openai_api_key'])
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Load embeddings from the most recent checkpoint
        self.embeddings_dict = self._load_latest_embeddings(embeddings_dir)
        
        # Prepare the embedding matrix and metadata
        self.poster_ids = list(self.embeddings_dict.keys())
        self.embedding_matrix = np.stack([
            self.embeddings_dict[poster_id]['embedding'] 
            for poster_id in self.poster_ids
        ])

    def _load_latest_embeddings(self, embeddings_dir: str) -> Dict:
        """Load the most recent embeddings checkpoint."""
        checkpoint_files = glob.glob(f"{embeddings_dir}/neurips_embeddings_*.pkl")
        if not checkpoint_files:
            raise FileNotFoundError("No embedding checkpoints found")
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with open(latest_checkpoint, 'rb') as f:
            return pickle.load(f)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for the search query."""
        query = query.replace("\n", " ")
        tokens = self.tokenizer.encode(query)
        if len(tokens) > 8192:  # Max token limit
            query = self.tokenizer.decode(tokens[:8192])
            
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for the k most relevant posters given a query."""
        query_embedding = self._get_query_embedding(query)
        similarities = np.dot(self.embedding_matrix, query_embedding)
        
        return self._get_top_k_results(similarities, k)

    def search_by_author(self, author_name: str, partial_match: bool = True) -> List[Dict]:
        """Search for posters by author name."""
        matching_posters = []
        author_name = author_name.lower()
        
        for poster_id, poster_data in self.embeddings_dict.items():
            authors = [author.lower() for author in poster_data['authors']]
            
            found_match = False
            if partial_match:
                found_match = any(author_name in author for author in authors)
            else:
                found_match = any(author_name == author for author in authors)
                
            if found_match:
                result = poster_data.copy()
                result.pop('embedding')
                matching_posters.append(result)
        
        # Sort by session and poster number
        matching_posters.sort(key=lambda x: (x['session_id'], x['poster_number']))
        return matching_posters

    def self_search(self, k: int = 5) -> List[Dict]:
        """Search for posters similar to self_description.txt."""
        try:
            with open('self_description.txt', 'r') as f:
                self_description = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("self_description.txt not found")
            
        return self.search(self_description, k)

    def _get_top_k_results(self, similarities: np.ndarray, k: int) -> List[Dict]:
        """Helper function to get top k results given similarity scores."""
        k = min(k, len(similarities))
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
        
        results = []
        for idx in top_k_indices:
            poster_id = self.poster_ids[idx]
            poster_data = self.embeddings_dict[poster_id].copy()
            poster_data['similarity_score'] = float(similarities[idx])
            poster_data.pop('embedding')
            results.append(poster_data)
            
        return results

    def format_results(self, results: List[Dict]) -> str:
        """Format search results for display."""
        if not results:
            return "No matching posters found."
            
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   Authors: {', '.join(result['authors'])}")
            output.append(f"   Session: {result['session_title']} (#{result['poster_number']})")
            output.append(f"   Location: {result['location']}")
            output.append(f"   URL: {result['url']}")
            if 'similarity_score' in result:
                output.append(f"   Similarity Score: {result['similarity_score']:.3f}")
            output.append("")
        
        return "\n".join(output)
    
if __name__ == "__main__":
    # Initialise the search engine
    searcher = PosterSearch()

    # Search for posters
    results = searcher.search("meta-learning reinforcement learning", k=5)

    # Search by author (partial match by default)
    author_results = searcher.search_by_author("Jakob Foerster")

    # Search by author with exact matching
    exact_author_results = searcher.search_by_author("Foerster, Jakob", partial_match=False)

    # Self search using self_description.txt
    self_results = searcher.self_search(k=10)

    # Display any of the results
    print(f"Concept search:")
    print('-'*50)
    print(searcher.format_results(results))
    print('\n\n\n')
    print("Author search:")
    print('-'*50)
    print(searcher.format_results(author_results))
    print('\n\n\n')
    print("Self search:")
    print('-'*50)
    print(searcher.format_results(self_results))