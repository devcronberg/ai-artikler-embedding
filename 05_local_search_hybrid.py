from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Load model
print("🚀 Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load all embeddings and texts
print("📚 Loading embeddings and corresponding text from 'chunks' directory...")
embeddings = []
filenames = []
texts = []
for txt_file in sorted(Path("chunks").glob("*.txt")):
    json_file = txt_file.with_suffix(".json")
    if json_file.exists():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        embeddings.append(data["embedding"])
        filenames.append(txt_file.name)
        texts.append(txt_file.read_text(encoding="utf-8"))

print(f"✅ Loaded {len(embeddings)} chunks.")

# Convert embeddings to tensor
embeddings_tensor = torch.tensor(embeddings)
print(f"✅ Converted embeddings to tensor with shape {embeddings_tensor.shape}")

while True:
    query = input("\n🔍 Enter your search (or 'exit' to quit): ")
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting.")
        break

    # First filter on keyword presence
    keyword = query.lower()
    filtered_indices = [i for i, text in enumerate(texts) if keyword in text.lower()]
    if not filtered_indices:
        print("⚠️ No chunks contain the keyword directly. Falling back to pure semantic search.")
        filtered_indices = list(range(len(embeddings)))  # use all

    # Create query embedding
    print(f"✍️ Creating embedding for your query: '{query}'")
    query_embedding = model.encode(query)
    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)

    # Compute cosine similarity only on filtered chunks
    filtered_embeddings = embeddings_tensor[filtered_indices]
    similarities = util.cos_sim(query_tensor, filtered_embeddings)[0]

    # Find top 5
    print("🔎 Calculating cosine similarity on filtered chunks...")
    top_values, top_pos = similarities.topk(min(5, len(filtered_indices)))

    print("🏆 Top 5 results:")
    for score, pos in zip(top_values.tolist(), top_pos.tolist()):
        actual_idx = filtered_indices[pos]
        snippet = texts[actual_idx][:100].replace("\n", " ")
        print(f"  - {filenames[actual_idx]} (similarity: {score:.3f})")
        print(f"    ✂️ '{snippet}...'")
