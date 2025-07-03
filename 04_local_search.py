from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Load model
print("🚀 Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load all embeddings from JSON files
print("📚 Loading embeddings from 'chunks' directory...")
embeddings = []
filenames = []
for json_file in sorted(Path("chunks").glob("*.json")):
    data = json.loads(json_file.read_text(encoding="utf-8"))
    embeddings.append(data["embedding"])
    filenames.append(data["filename"])
print(f"✅ Loaded {len(embeddings)} chunks.")

# Convert to tensor
embeddings_tensor = torch.tensor(embeddings)
print(f"✅ Converted embeddings to tensor with shape {embeddings_tensor.shape}")

# Loop: user can keep searching
while True:
    query = input("\n🔍 Enter your search (or 'exit' to stop): ")
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting.")
        break

    print(f"✍️ Creating embedding for your query: '{query}'")
    query_embedding = model.encode(query)
    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)

    print("🔎 Calculating cosine similarity...")
    similarities = util.cos_sim(query_tensor, embeddings_tensor)[0]

    print("🏆 Top 5 results:")
    top_values, top_indices = similarities.topk(5)
    for score, idx in zip(top_values.tolist(), top_indices.tolist()):
        print(f"  - {filenames[idx]} (similarity: {score:.3f})")
