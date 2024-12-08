"""
author_search.py - Tools for analyzing and visualizing author relationships at NeurIPS.

This module provides functionality to analyze relationships between authors based on their papers,
including finding similar authors, visualizing similarity matrices, and identifying potential
collaborations. It uses pre-computed embeddings and similarity matrices.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import pickle

class AuthorSearch:
    def __init__(self, embeddings_path: str = 'multi_paper_author_embeddings.pkl',
                 similarity_matrix_path: Optional[str] = None):
        """
        Initialize the author search system.

        Args:
            embeddings_path: Path to the pre-computed author embeddings file
            similarity_matrix_path: Optional path to pre-computed similarity matrix
        """
        # Load author data
        with open(embeddings_path, 'rb') as f:
            self.author_data = pickle.load(f)
        
        self.authors = list(self.author_data.keys())
        
        # Load or compute similarity matrix
        if similarity_matrix_path:
            self.similarity_df = pd.read_pickle(similarity_matrix_path)
        else:
            embedding_matrix = np.stack([
                self.author_data[author]['embedding'] 
                for author in self.authors
            ])
            similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
            self.similarity_df = pd.DataFrame(
                similarity_matrix,
                index=self.authors,
                columns=self.authors
            )

    def find_similar_authors(self, author_name: str, top_k: int = 5,
                           exclude_coauthors: bool = False) -> List[Dict]:
        """
        Find the most similar authors to a given author.

        Args:
            author_name: Name of the author to analyze
            top_k: Number of similar authors to return
            exclude_coauthors: Whether to exclude co-authors from results

        Returns:
            List of dictionaries containing similar author information
        """
        if author_name not in self.authors:
            raise ValueError(f"Author '{author_name}' not found in database")
        
        # Get similarities
        similarities = self.similarity_df[author_name]
        
        # Filter out the author themselves and sort
        similar_authors = similarities.drop(author_name).sort_values(ascending=False)
        
        # Optionally filter out co-authors
        if exclude_coauthors:
            coauthors = set(
                author for paper in self.author_data[author_name]['papers']
                for author in paper.get('authors', [])
            ) - {author_name}
            similar_authors = similar_authors.drop(list(coauthors), errors='ignore')
        
        # Get top k results
        results = []
        for author, similarity in similar_authors[:top_k].items():
            results.append({
                'name': author,
                'similarity': similarity,
                'paper_count': self.author_data[author]['paper_count'],
                'papers': self.author_data[author]['papers']
            })
        
        return results

    def plot_similarity_matrix(self, top_n: int = 50, figsize: Tuple[int, int] = (15, 15)):
        """
        Plot similarity matrix for top N authors by paper count.

        Args:
            top_n: Number of top authors to include
            figsize: Figure size as (width, height)
        """
        # Get top N authors by paper count
        top_authors = sorted(
            self.authors,
            key=lambda x: self.author_data[x]['paper_count'],
            reverse=True
        )[:top_n]
        
        # Extract submatrix
        top_similarity = self.similarity_df.loc[top_authors, top_authors]
        
        # Create labels with paper counts
        labels = [
            f"{author} ({self.author_data[author]['paper_count']})"
            for author in top_authors
        ]
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            top_similarity,
            xticklabels=labels,
            yticklabels=labels,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title(f'Author Similarity Matrix (Top {top_n} Authors by Paper Count)')
        plt.tight_layout()
        return plt.gcf()

    def format_results(self, results: List[Dict], result_type: str = 'similar_authors') -> str:
        """
        Format search results for display.

        Args:
            results: List of result dictionaries
            result_type: Type of results ('similar_authors' or 'similar_pairs')

        Returns:
            Formatted string of results
        """
        if not results:
            return "No results found."
            
        output = []
        if result_type == 'similar_authors':
            for i, result in enumerate(results, 1):
                output.append(f"{i}. {result['name']} (similarity: {result['similarity']:.3f})")
                output.append(f"   Papers: {result['paper_count']}")
                for paper in result['papers'][:2]:
                    output.append(f"   - {paper['title']}")
                output.append("")
        elif result_type == 'similar_pairs':
            for i, pair in enumerate(results, 1):
                output.append(f"{i}. Similarity: {pair['similarity']:.3f}")
                output.append(f"\nAuthor 1: {pair['author1']['name']}")
                for paper in pair['author1']['papers']:
                    output.append(f"- {paper['title']}")
                output.append(f"\nAuthor 2: {pair['author2']['name']}")
                for paper in pair['author2']['papers']:
                    output.append(f"- {paper['title']}")
                output.append("-" * 80)
        
        return "\n".join(output)

def load_author_search(embeddings_path: str = 'multi_paper_author_embeddings.pkl',
                      similarity_matrix_path: Optional[str] = None) -> AuthorSearch:
    """
    Convenience function to load the author search system.
    
    Args:
        embeddings_path: Path to author embeddings file
        similarity_matrix_path: Optional path to pre-computed similarity matrix
        
    Returns:
        Initialized AuthorSearch instance
    """
    return AuthorSearch(embeddings_path, similarity_matrix_path)

# Example usage
if __name__ == "__main__":
    # Initialise search
    search = load_author_search()
    
    # Find similar authors
    author = "Philip Torr"
    similar = search.find_similar_authors(author, top_k=5, exclude_coauthors=True)
    print(search.format_results(similar))
    
    
    # Plot similarity matrix
    search.plot_similarity_matrix(top_n=30)
    plt.show()