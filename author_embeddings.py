import os
import json
import pickle
import yaml
import time
import random
from collections import defaultdict
from typing import Dict, List, Set
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import tiktoken
from concurrent.futures import ThreadPoolExecutor

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 10
MAX_WORKERS = 5
MAX_TOKENS = 8192
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120
OUTPUT_FILE = "multi_paper_author_embeddings.pkl"
MIN_PAPERS = 2

# Initialize OpenAI client
config = yaml.safe_load(open('config.yaml', 'r'))
client = OpenAI(api_key=config['openai_api_key'])
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_multi_paper_authors() -> Dict[str, List[Dict]]:
    """Create mapping of authors with multiple papers to their papers."""
    with open('neurips_sessions.json', 'r') as f:
        sessions_data = json.load(f)

    author_papers = defaultdict(list)
    total_papers = 0
    
    # First pass: collect all papers for each author
    for session_id, session_data in sessions_data.items():
        for poster in session_data['posters']:
            total_papers += 1
            for author in poster['authors']:
                author_papers[author].append({
                    'title': poster['title'],
                    'abstract': poster['abstract'],
                    'session_id': session_id,
                    'poster_number': poster['poster_number']
                })
    
    # Second pass: filter for authors with multiple papers
    multi_paper_authors = {
        author: papers for author, papers in author_papers.items()
        if len(papers) >= MIN_PAPERS
    }
    
    print("\nAuthor Analysis:")
    print(f"Total papers in dataset: {total_papers}")
    print(f"Total unique authors: {len(author_papers)}")
    print(f"Authors with multiple papers: {len(multi_paper_authors)}")
    
    # Analyze paper distribution for multi-paper authors
    paper_counts = [len(papers) for papers in multi_paper_authors.values()]
    print("\nMulti-paper author statistics:")
    print(f"Maximum papers per author: {max(paper_counts)}")
    print(f"Average papers per author: {np.mean(paper_counts):.2f}")
    print(f"Median papers per author: {np.median(paper_counts):.1f}")
    
    # Print some examples of prolific authors
    top_authors = sorted(
        multi_paper_authors.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]
    
    print("\nMost prolific authors:")
    for author, papers in top_authors:
        print(f"{author}: {len(papers)} papers")
    
    return multi_paper_authors

def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to maximum token length."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Get embedding with retry logic."""
    text = text.replace("\n", " ")
    text = truncate_text(text)
    
    retry_delay = INITIAL_RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            embedding = client.embeddings.create(
                input=[text],
                model=model,
                encoding_format="float"
            ).data[0].embedding
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"\nFailed to get embedding after {MAX_RETRIES} attempts: {str(e)}")
                raise
            
            retry_delay = min(MAX_RETRY_DELAY, retry_delay * 2)
            jitter = random.uniform(0, 0.1 * retry_delay)
            time.sleep(retry_delay + jitter)
    
    raise Exception("Failed to get embedding")

def process_author_batch(author_batch: List[tuple]) -> Dict[str, Dict]:
    """Process a batch of authors to get their embeddings."""
    results = {}
    
    for author, papers in author_batch:
        # Concatenate all paper titles and abstracts
        combined_text = " ".join([
            f"Title: {paper['title']} Abstract: {paper['abstract']}"
            for paper in papers
        ])
        
        try:
            embedding = get_embedding(combined_text)
            results[author] = {
                'embedding': embedding,
                'paper_count': len(papers),
                'papers': papers
            }
        except Exception as e:
            print(f"\nError processing author {author}: {str(e)}")
    
    return results

def main():
    # Get mapping of authors with multiple papers
    author_papers = get_multi_paper_authors()
    
    # Convert to list of (author, papers) tuples for batch processing
    author_items = list(author_papers.items())
    
    # Initialize results dictionary
    author_embeddings = {}
    
    # Process authors in batches
    print("\nGenerating author embeddings...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(author_items), BATCH_SIZE):
            batch = author_items[i:i + BATCH_SIZE]
            futures.append(executor.submit(process_author_batch, batch))
        
        with tqdm(total=len(author_items), desc="Processing authors") as pbar:
            for future in futures:
                try:
                    batch_results = future.result()
                    author_embeddings.update(batch_results)
                    pbar.update(len(batch_results))
                except Exception as e:
                    print(f"\nBatch processing error: {str(e)}")
    
    # Save results
    print("\nSaving author embeddings...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(author_embeddings, f)
    
    print("\nProcessing complete:")
    print(f"Total authors processed: {len(author_embeddings)}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()