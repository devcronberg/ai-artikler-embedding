from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Find all text files
chunk_dir = Path("chunks")
chunk_files = list(chunk_dir.glob("*.txt"))

print(f"Found {len(chunk_files)} chunks...")

for i, chunk_file in enumerate(chunk_files, start=1):
    # Read text
    text = chunk_file.read_text(encoding="utf-8")
    
    # Create embedding
    embedding = model.encode(text).tolist()  # convert to normal Python list

    # Save to json file
    json_file = chunk_file.with_suffix(".json")
    with json_file.open("w", encoding="utf-8") as f:
        json.dump({
            "filename": chunk_file.name,
            "embedding": embedding
        }, f)
    
    # Progress indicator
    if i % 50 == 0 or i == len(chunk_files):
        print(f"✅ {i}/{len(chunk_files)} done ({chunk_file.name})")

print("🎉 Done! All embeddings saved as .json files in the 'chunks' directory.")
