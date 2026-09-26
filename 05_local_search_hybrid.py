# Trin 5: Et alternativ til trin 4, der kombinerer ordret søgning med
# søgning efter betydning. Kombinationen kaldes her "hybrid søgning".
# Kør trin 1-3 først. Der skal være mindst ét tekststykke med en tilhørende
# JSON-fil med embedding. Start programmet fra projektets mappe.
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Brug samme færdigtrænede model som i trin 3. Søgningen foregår lokalt,
# efter at modellen er hentet fra internettet første gang.
print("🚀 Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Indlæs både teksterne og deres talrepræsentationer (embeddings).
# Et tekststykke springes over, hvis det endnu ikke har en JSON-fil.
# Samme plads i de tre lister hører altid til samme tekststykke.
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

# En tensor er her en tabel med én række tal pr. tekststykke.
# Torch bruger den til at udføre sammenligningerne effektivt.
embeddings_tensor = torch.tensor(embeddings)
print(f"✅ Converted embeddings to tensor with shape {embeddings_tensor.shape}")

# Søg flere gange uden at indlæse model og filer på ny.
while True:
    query = input("\n🔍 Enter your search (or 'exit' to quit): ")
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting.")
        break

    # Behold først kun tekster, hvor HELE søgeteksten optræder som en delstreng.
    # Store og små bogstaver behandles ens. Ordene søges ikke enkeltvis.
    keyword = query.lower()
    filtered_indices = [i for i, text in enumerate(texts) if keyword in text.lower()]
    if not filtered_indices:
        print("⚠️ No chunks contain the keyword directly. Falling back to pure semantic search.")
        # Uden ordrette træffere søger vi efter betydning i alle tekststykker.
        filtered_indices = list(range(len(embeddings)))

    # Omdan spørgsmålet til samme slags tal som tekststykkerne.
    print(f"✍️ Creating embedding for your query: '{query}'")
    query_embedding = model.encode(query)
    # Giv spørgsmålets tal form som en tabel med én række.
    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)

    # Sammenlign kun med de udvalgte tekster. Cosinuslighed måler, hvor ens
    # talmønstrenes retninger er. Højere score betyder større lighed, ikke sikkerhed.
    filtered_embeddings = embeddings_tensor[filtered_indices]
    similarities = util.cos_sim(query_tensor, filtered_embeddings)[0]

    # Vis højst fem resultater, eller færre hvis filteret fandt færre tekster.
    print("🔎 Calculating cosine similarity on filtered chunks...")
    top_values, top_pos = similarities.topk(min(5, len(filtered_indices)))

    print("🏆 Top 5 results:")
    for score, pos in zip(top_values.tolist(), top_pos.tolist()):
        # Oversæt pladsen i den filtrerede liste til pladsen i den oprindelige liste.
        actual_idx = filtered_indices[pos]
        # Vis de første 100 tegn, så man kan få et hurtigt indtryk af indholdet.
        snippet = texts[actual_idx][:100].replace("\n", " ")
        print(f"  - {filenames[actual_idx]} (similarity: {score:.3f})")
        print(f"    ✂️ '{snippet}...'")
