# Trin 2: Del den lange tekst fra trin 1 i små tekststykker kaldet "chunks".
# Små stykker gør det muligt at finde relevante passager i stedet for at
# behandle hele manualen som ét søgeresultat. Her bruges ingen AI-model.
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Læs hele den rensede tekstfil. Kør programmet fra projektets mappe.
text = Path("hp34c-ohpg-en-full-clean.txt").read_text(encoding="utf-8")

# Opdel først ved naturlige skel som afsnit og mellemrum, når det er muligt.
# Størrelsen måles i tegn, ikke ord. Hvert stykke er højst 500 tegn.
# Et ønsket overlap på 50 tegn gentager lidt tekst mellem nabostykker,
# så sammenhængen ved en grænse ikke så let går tabt.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
print(f"Total number of chunks: {len(chunks)}")

# Opret mappen, hvis den mangler, og gem ét tekststykke pr. fil.
# Eksisterende filer med samme navn overskrives; andre gamle filer slettes ikke.
output_dir = Path("chunks")
output_dir.mkdir(exist_ok=True)

# Nummereringen giver navne som chunk_001.txt, chunk_002.txt osv.
for i, chunk in enumerate(chunks, start=1):
    chunk_file = output_dir / f"chunk_{i:03}.txt"
    chunk_file.write_text(chunk, encoding="utf-8")
    print(f"Saved {chunk_file}")

print("✅ Done! All chunks saved in the 'chunks' directory.")
