import os
import json
import pickle
import yaml
import time
import random
import glob
import numpy as np
from typing import Dict, Optional, Tuple, List, Set
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import tiktoken

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 10
SAVE_INTERVAL = 100
OUTPUT_DIR = "embeddings"
MAX_WORKERS = 5
MAX_TOKENS = 8192
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120

# Initialize OpenAI client
config = yaml.safe_load(open('config.yaml', 'r'))
client = OpenAI(api_key=config['openai_api_key'])
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_unique_key(session_id: str, poster_number: int) -> str:
    """Create a unique key combining session ID and poster number."""
    return f"{session_id}_{poster_number}"

def get_all_posters(sessions_data: Dict) -> List[Dict]:
    """Extract all posters from all sessions into a flat list with session information."""
    all_posters = []
    print("\nSession breakdown:")
    for session_id, session_data in sessions_data.items():
        session_posters = session_data['posters']
        print(f"Session {session_id}: {len(session_posters)} posters")
        for poster in session_posters:
            poster_with_session = poster.copy()
            poster_with_session['session_id'] = session_id
            poster_with_session['session_title'] = session_data['session_info']['session_title']
            poster_with_session['location'] = session_data['session_info']['location']
            poster_with_session['unique_key'] = get_unique_key(session_id, poster['poster_number'])
            all_posters.append(poster_with_session)
    
    all_posters.sort(key=lambda x: (x['session_id'], x['poster_number']))
    return all_posters

def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to maximum token length."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Get embedding with retry logic."""
    if not text or text.strip() == "":
        return None
        
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
                print(f"Failed to get embedding: {str(e)}")
                return None
            
            retry_delay = min(MAX_RETRY_DELAY, retry_delay * 2)
            jitter = random.uniform(0, 0.1 * retry_delay)
            sleep_time = retry_delay + jitter
            time.sleep(sleep_time)
    
    return None

def process_batch(posters: List[Dict]) -> Dict[str, Dict]:
    """Process a batch of posters to get their embeddings."""
    embeddings = {}
    
    for poster in posters:
        combined_text = f"Title: {poster['title']} Abstract: {poster['abstract']}"
        embedding = get_embedding(combined_text)
        
        if embedding is not None:
            embeddings[poster['unique_key']] = {
                'embedding': embedding,
                'title': poster['title'],
                'authors': poster['authors'],
                'url': poster['url'],
                'session_id': poster['session_id'],
                'session_title': poster['session_title'],
                'location': poster['location'],
                'poster_number': poster['poster_number']
            }
    
    return embeddings

def save_embeddings(embeddings: Dict, processed_count: int) -> str:
    """Save embeddings to checkpoint file."""
    checkpoint_file = f"{OUTPUT_DIR}/neurips_embeddings_{processed_count}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(embeddings, f)
    return checkpoint_file

def find_last_checkpoint() -> Tuple[Dict, Set[str]]:
    """Find and load the most recent checkpoint."""
    checkpoint_files = glob.glob(f"{OUTPUT_DIR}/neurips_embeddings_*.pkl")
    if not checkpoint_files:
        return {}, set()
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    with open(latest_checkpoint, 'rb') as f:
        embeddings = pickle.load(f)
    
    processed_keys = set(embeddings.keys())
    print("\nCheckpoint analysis:")
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"Current embeddings: {len(embeddings)}")
    
    return embeddings, processed_keys

def cleanup_old_checkpoints(keep_file: str):
    """Remove old checkpoint files."""
    checkpoint_files = glob.glob(f"{OUTPUT_DIR}/neurips_embeddings_*.pkl")
    for file in checkpoint_files:
        if file != keep_file and not file.endswith('_final.pkl'):
            os.remove(file)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all sessions data
    with open('neurips_sessions.json', 'r') as f:
        sessions_data = json.load(f)
    
    # Get all posters from all sessions
    all_posters = get_all_posters(sessions_data)
    total_posters = len(all_posters)
    actual_poster_keys = {poster['unique_key'] for poster in all_posters}
    print(f"Found {total_posters} total posters across all sessions")
    
    # Load last checkpoint and identify remaining work
    all_embeddings, processed_keys = find_last_checkpoint()
    
    # Filter out already processed posters
    remaining_posters = [p for p in all_posters if p['unique_key'] in actual_poster_keys - processed_keys]
    print(f"Remaining posters to process: {len(remaining_posters)}")
    
    if not remaining_posters:
        print("All posters have been processed. Nothing to do.")
        return
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {}
        for i in range(0, len(remaining_posters), BATCH_SIZE):
            batch = remaining_posters[i:i + BATCH_SIZE]
            future = executor.submit(process_batch, batch)
            future_to_batch[future] = i
        
        with tqdm(total=len(remaining_posters), desc="Processing posters") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_embeddings = future.result()
                    all_embeddings.update(batch_embeddings)
                    pbar.update(len(batch_embeddings))
                    
                    if len(all_embeddings) % SAVE_INTERVAL == 0:
                        checkpoint_file = save_embeddings(all_embeddings, len(all_embeddings))
                        print(f"\nSaved checkpoint with {len(all_embeddings)} embeddings")
                        cleanup_old_checkpoints(checkpoint_file)
                
                except Exception as e:
                    batch_index = future_to_batch[future]
                    print(f"\nError processing batch starting at index {batch_index}: {e}")
    
    # Save final embeddings
    final_file = save_embeddings(all_embeddings, len(all_embeddings))
    cleanup_old_checkpoints(final_file)
    print(f"\nProcessing complete:")
    print(f"Total posters: {total_posters}")
    print(f"Successfully embedded: {len(all_embeddings)}")

if __name__ == "__main__":
    main()