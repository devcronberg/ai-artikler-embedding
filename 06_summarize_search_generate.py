from pathlib import Path
import json
import torch
from sentence_transformers import SentenceTransformer, util
from llm_utils import ask_llm

# Load local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load embeddings
print("📚 Loading embeddings and text snippets...")
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

embeddings_tensor = torch.tensor(embeddings)
print(f"✅ Loaded {len(embeddings)} chunks.")

# Start conversation history
conversation = [
    {"role": "system", "content": "You are a helpful expert on the HP-34C calculator manual. Provide clear, precise answers."}
]

while True:
    user_query = input("\n🔍 What would you like to ask? (or 'exit'): ")
    if user_query.lower() in ("exit", "quit"):
        break

    # 1️⃣ summarize (always in English, optimized for search)
    summary_prompt = conversation + [{
        "role": "user",
        "content": f"Summarize this question into a very short, search-optimized phrase (under 10 words), in English, using technical keywords if possible: '{user_query}'"
    }]
    summary = ask_llm(summary_prompt).strip()
    print(f"📝 Optimized English summary for search: {summary}")

    # 2️⃣ local semantic search
    query_embedding = model.encode(summary)
    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)
    similarities = util.cos_sim(query_tensor, embeddings_tensor)[0]
    top_values, top_indices = similarities.topk(5)

    top_chunks = []
    for score, idx in zip(top_values.tolist(), top_indices.tolist()):
        print(f"  - {filenames[idx]} (similarity: {score:.3f})")
        snippet = texts[idx][:200].replace("\n", " ")
        print(f"    ✂️ '{snippet}...'")
        top_chunks.append(texts[idx])

    # 3️⃣ final question to LLM with context
    context_text = "\n---\n".join(top_chunks)
    final_prompt = conversation + [
        {"role": "user", "content": f"""\
Here are the most relevant excerpts from the HP-34C manual:

{context_text}

Based on these, please answer this question: "{user_query}"

Try to respond in the same language the question was asked in.
"""}
    ]
    answer = ask_llm(final_prompt).strip()
    print(f"\n💬 Answer from LLM:\n{answer}")

    # Update conversation history
    conversation.append({"role": "user", "content": user_query})
    conversation.append({"role": "assistant", "content": answer})
