# Trin 3: Lav en "embedding" for hvert tekststykke fra trin 2.
# En embedding er en liste af tal, der repræsenterer tekstens indhold.
# Tekster med lignende betydning får ofte lignende talmønstre. Det gør
# senere søgning mulig uden at kræve præcis de samme ord i tekst og spørgsmål.
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

# Indlæs en færdigtrænet model; vi træner ikke vores egen model her.
# Første gang hentes modellen fra internettet. Derefter bruges den lokalt.
# Søgeprogrammerne skal bruge samme model, så tallene kan sammenlignes.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Find tekststykkerne i mappen fra trin 2. Kør fra projektets mappe.
chunk_dir = Path("chunks")
chunk_files = list(chunk_dir.glob("*.txt"))

print(f"Found {len(chunk_files)} chunks...")

for i, chunk_file in enumerate(chunk_files, start=1):
    # Læs ét tekststykke ad gangen.
    text = chunk_file.read_text(encoding="utf-8")
    
    # Modellen omdanner teksten til tal, ikke til et resumé eller et svar.
    # En almindelig Python-liste kan gemmes i JSON-formatet nedenfor.
    embedding = model.encode(text).tolist()

    # Gem tal og kildefilens navn ved siden af teksten, fx chunk_001.json.
    # JSON er et struktureret tekstformat. En eksisterende fil overskrives.
    json_file = chunk_file.with_suffix(".json")
    with json_file.open("w", encoding="utf-8") as f:
        json.dump({
            "filename": chunk_file.name,
            "embedding": embedding
        }, f)
    
    # Vis fremdrift for hver 50 tekststykker og ved det sidste stykke.
    if i % 50 == 0 or i == len(chunk_files):
        print(f"✅ {i}/{len(chunk_files)} done ({chunk_file.name})")

print("🎉 Done! All embeddings saved as .json files in the 'chunks' directory.")
