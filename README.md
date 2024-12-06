# NeurIPS 2024 Poster Search

This repository provides tools to search through NeurIPS 2024 poster sessions using semantic search and author matching. The search functionality uses OpenAI's `text-embedding-3-small` model to find posters related to your research interests.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/neurips-poster-search
cd neurips-poster-search

# Install requirements
pip install -r requirements.txt

# Create and edit your config.yaml
cp config_template.yaml config.yaml
# Edit config.yaml with your OpenAI API key

# Edit self_description.txt with your research interests
nano self_description.txt

# Run the search
python search.py
```

## Setup Details

1. Install dependencies from `requirements.txt`
2. Create `config.yaml` with your OpenAI API key:
   ```yaml
   openai_api_key: "your-api-key-here"
   ```
3. Edit `self_description.txt` to describe your research interests (example provided in the file)

## Usage

The repository includes three main scripts:

1. `search.py`: The main search interface
   ```bash
   python search.py
   ```

2. `all_posters.py`: (Optional) Scrape poster data from NeurIPS website
   - Requires additional configuration in config.yaml for NeurIPS credentials
   - Only needed if you want to update the poster database

3. `embeddings.py`: (Optional) Generate embeddings for poster data
   - Only needed if you've updated the poster database

## Search Features

- Semantic search based on research interests
- Author-based search
- "Self search" using your research description

## Data

- `neurips_sessions.json`: Contains all poster data
- `embeddings/`: Contains pre-computed embeddings
- `self_description.txt`: Your research interests for personalised search

## Optional: Updating Poster Data

If you want to update the poster database:

1. Add NeurIPS credentials to config.yaml:
   ```yaml
   username: "your-neurips-username"
   password: "your-neurips-password"
   ```

2. Run the scraper:
   ```bash
   python all_posters.py
   ```

3. Generate new embeddings:
   ```bash
   python embeddings.py
   ```

## Note

This tool is provided for research purposes to help conference attendees find relevant posters. Please be mindful of NeurIPS's terms of service when using the scraper functionality.

## License

MIT License