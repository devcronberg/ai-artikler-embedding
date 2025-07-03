from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load text file
text = Path("hp34c-ohpg-en-full-clean.txt").read_text(encoding="utf-8")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
print(f"Total number of chunks: {len(chunks)}")

# Save each chunk to a separate file
output_dir = Path("chunks")
output_dir.mkdir(exist_ok=True)

for i, chunk in enumerate(chunks, start=1):
    chunk_file = output_dir / f"chunk_{i:03}.txt"
    chunk_file.write_text(chunk, encoding="utf-8")
    print(f"Saved {chunk_file}")

print("✅ Done! All chunks saved in the 'chunks' directory.")
