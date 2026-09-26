# Trin 4: Find tekststykker, der ligner dit spørgsmål i betydning.
# Dette kaldes semantisk søgning. Programmet finder filer, men skriver ikke et svar.
# Kør trin 1-3 først og start programmet fra projektets mappe.
# Der skal være mindst fem embeddings, fordi programmet vælger fem resultater.
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Brug samme model som i trin 3, så spørgsmål og tekst får sammenlignelige tal.
# Modellen hentes ved behov fra internettet; selve søgningen foregår lokalt.
print("🚀 Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Indlæs de gemte talrepræsentationer (embeddings), så de ikke skal beregnes igen.
# De to lister følges ad: samme plads indeholder tal og filnavn for samme tekst.
print("📚 Loading embeddings from 'chunks' directory...")
embeddings = []
filenames = []
for json_file in sorted(Path("chunks").glob("*.json")):
    data = json.loads(json_file.read_text(encoding="utf-8"))
    embeddings.append(data["embedding"])
    filenames.append(data["filename"])
print(f"✅ Loaded {len(embeddings)} chunks.")

# En tensor er her en tabel med tal: én række pr. tekststykke.
# Torch bruger tabellen til at sammenligne mange tekststykker på én gang.
embeddings_tensor = torch.tensor(embeddings)
print(f"✅ Converted embeddings to tensor with shape {embeddings_tensor.shape}")

# Gentag søgningen, indtil brugeren skriver exit eller quit.
while True:
    query = input("\n🔍 Enter your search (or 'exit' to stop): ")
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting.")
        break

    print(f"✍️ Creating embedding for your query: '{query}'")
    # Lav også spørgsmålet om til tal. unsqueeze(0) giver tabellen én række.
    query_embedding = model.encode(query)
    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)

    print("🔎 Calculating cosine similarity...")
    # Cosinuslighed sammenligner talmønstrenes retning. Højere betyder mere ens.
    # Scoren er ikke en procent eller en garanti for, at teksten besvarer spørgsmålet.
    # [0] vælger scorerne for det ene spørgsmål, vi netop har indtastet.
    similarities = util.cos_sim(query_tensor, embeddings_tensor)[0]

    print("🏆 Top 5 results:")
    # Find de fem højeste scorer og deres pladser i listen med filnavne.
    top_values, top_indices = similarities.topk(5)
    for score, idx in zip(top_values.tolist(), top_indices.tolist()):
        print(f"  - {filenames[idx]} (similarity: {score:.3f})")
