# Trin 6: Find relevante tekststykker, og lad en sprogmodel formulere et svar.
# En sprogmodel (LLM) er AI, der kan læse og skrive tekst.
# Denne kombination af søgning og svarskrivning kaldes ofte RAG.
# Kør trin 1-3 først; trin 4 og 5 er valgfrie demonstrationer af søgningen.
# Der skal være mindst fem tekststykker med embeddings. Kør fra projektets mappe.
# Vigtigt: Spørgsmål, samtalehistorik og udvalgte tekststykker sendes til
# OpenRouter og den valgte modeludbyder via llm_utils. Brug ikke fortrolige data.
from pathlib import Path
import json
import torch
from sentence_transformers import SentenceTransformer, util
from llm_utils import ask_llm

# Den lokale model laver tal til søgning, ikke selve svaret.
# Den skal være den samme som i trin 3 og hentes fra internettet ved behov.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Indlæs tekst og talrepræsentation for hvert tekststykke, der har begge filer.
# De tre lister bruger samme rækkefølge, så et søgeresultat kan kobles til teksten.
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

# Saml tallene i en tabel (tensor), som Torch kan regne på.
embeddings_tensor = torch.tensor(embeddings)
print(f"✅ Loaded {len(embeddings)} chunks.")

# Samtalens beskeder har roller: system giver grundinstruktionen,
# user er brugerens spørgsmål, og assistant er modellens svar.
# Historikken lever kun i hukommelsen, indtil programmet afsluttes.
conversation = [
    {"role": "system", "content": "You are a helpful expert on the HP-34C calculator manual. Provide clear, precise answers."}
]

while True:
    user_query = input("\n🔍 What would you like to ask? (or 'exit'): ")
    if user_query.lower() in ("exit", "quit"):
        break

    # Første eksterne AI-kald: Bed om en kort engelsk søgeformulering,
    # fordi manualen er på engelsk. En "prompt" er instruktionen til modellen.
    # Historikken sendes også med, så tidligere spørgsmål kan give sammenhæng.
    # Modellen bliver bedt om under 10 ord, men det kontrolleres ikke i koden.
    summary_prompt = conversation + [{
        "role": "user",
        "content": f"Summarize this question into a very short, search-optimized phrase (under 10 words), in English, using technical keywords if possible: '{user_query}'"
    }]
    summary = ask_llm(summary_prompt).strip()
    print(f"📝 Optimized English summary for search: {summary}")

    # Lokal søgning: Lav søgeformuleringen om til tal, og find de fem
    # tekststykker med størst cosinuslighed, dvs. lignende retning i talmønstret.
    # Scoren måler lighed, ikke sandsynligheden for et korrekt svar.
    query_embedding = model.encode(summary)
    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)
    similarities = util.cos_sim(query_tensor, embeddings_tensor)[0]
    top_values, top_indices = similarities.topk(5)

    # Skærmen viser kun et kort uddrag; hele tekststykket gemmes til modellen.
    top_chunks = []
    for score, idx in zip(top_values.tolist(), top_indices.tolist()):
        print(f"  - {filenames[idx]} (similarity: {score:.3f})")
        snippet = texts[idx][:200].replace("\n", " ")
        print(f"    ✂️ '{snippet}...'")
        top_chunks.append(texts[idx])

    # Andet eksterne AI-kald: Send spørgsmålet, historikken og de fundne
    # tekststykker som baggrund ("kontekst"). Hele manualen sendes ikke med.
    # Modellen skriver et nyt svar ud fra materialet, men kan stadig tage fejl.
    # Kontrollér derfor vigtige oplysninger i manualen.
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

    # Gem spørgsmål og svar til næste runde. De fundne tekststykker gemmes
    # ikke i historikken. En lang samtale sender gradvist mere tekst til tjenesten.
    conversation.append({"role": "user", "content": user_query})
    conversation.append({"role": "assistant", "content": answer})
