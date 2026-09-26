# Eksempel på embedding

Python filer beskrevet i artiklen: https://mcronberg.github.io/aiartikler/embeddings2/

## Hvad gør projektet?

Projektet gør det muligt at søge i en PDF-manual til HP-34C og få en
sprogmodel til at formulere svar ud fra relevante uddrag. Modellerne er
færdigtrænede; projektet træner ikke sin egen AI.

## Filernes rækkefølge

Kør kommandoerne fra projektets mappe. Installer først pakkerne:

```powershell
python -m pip install -r requriments.txt
```

1. [01_extract_txt_from_pdf.py](01_extract_txt_from_pdf.py) læser manualen og gemmer en renset tekstfil. PDF'en skal indeholde tekst, ikke kun scannede billeder.
2. [02_split_txt_to_chunks.py](02_split_txt_to_chunks.py) deler teksten i små tekststykker i mappen `chunks`.
3. [03_create_embeddings.py](03_create_embeddings.py) omdanner hvert tekststykke til en liste af tal og gemmer den i en JSON-fil ved siden af teksten.
4. [04_local_search.py](04_local_search.py) finder de fem tekststykker, der minder mest om søgningen i betydning, og viser deres filnavne.
5. [05_local_search_hybrid.py](05_local_search_hybrid.py) er en alternativ søgning: Den filtrerer først på hele søgeteksten som ordret delstreng og sorterer derefter efter betydning. Uden ordrette træffere søger den i alle tekststykker.
6. [06_summarize_search_generate.py](06_summarize_search_generate.py) finder relevante tekststykker og bruger en ekstern sprogmodel til at formulere et svar.

Start fx første trin med `python 01_extract_txt_from_pdf.py`. Kør trin 1-3
i rækkefølge; derefter kan du vælge mellem trin 4, 5 og 6. Trin 4 og 6
kræver mindst fem tekststykker med embeddings. Skriv `exit` for at afslutte søgningen.

[llm_utils.py](llm_utils.py) er hjælpefilen, som trin 6 bruger til at kontakte
sprogmodellen. [requriments.txt](requriments.txt) forklarer, hvilke ekstra
Python-pakker projektet bruger.

## Begreber uden AI-forudsætninger

- **Chunk:** Et lille tekststykke fra manualen. Her bruges højst 500 tegn med et ønsket overlap på 50 tegn mellem nabostykker for at bevare sammenhæng.
- **Embedding:** En liste af tal, der repræsenterer en teksts indhold. Tekster med lignende betydning får ofte lignende talmønstre. Det er ikke et resumé.
- **Semantisk søgning:** Søgning efter betydning, så spørgsmål og manual ikke behøver bruge præcis de samme ord.
- **Lighedsscore:** Et mål for, hvor ens to talmønstre er. En høj score er ikke en procentvis sikkerhed for et korrekt svar.
- **Sprogmodel (LLM):** En AI-model, der kan formulere tekst. Den er forskellig fra modellen, der laver embeddings.
- **RAG:** At finde relevante kildetekster først og give dem til en sprogmodel som baggrund for dens svar.

## Lokalt og eksternt

Trin 1-5 behandler manualen lokalt. Embedding-modellen hentes fra internettet
første gang og skal være den samme, når tekster og søgninger omdannes til tal.

Trin 6 kræver internet og en OpenRouter-nøgle i miljøvariablen
`OPENROUTER_API_KEY` eller en lokal `.env`-fil. Hold nøglen hemmelig, og
medtag aldrig filen med nøglen i Git. Trin 6 sender spørgsmål og
samtalehistorik til OpenRouter og den valgte modeludbyder samt udvalgte
manualuddrag, når svaret skal skrives. Brug derfor ikke fortroligt materiale.
Modellers tilgængelighed, priser og brugsgrænser kan ændre sig.

Svar kan være forkerte, selv om de bygger på manualen. Kontrollér vigtige
oplysninger i originalen. Samtalehistorikken gemmes kun, mens programmet kører.

De genererede tekst- og JSON-filer er data og får ikke kodekommentarer,
da kommentarerne ellers ville blive en del af det materiale, der søges i.
Ved genkørsel overskrives filer med samme navn, men gamle overskydende
chunks slettes ikke automatisk.
