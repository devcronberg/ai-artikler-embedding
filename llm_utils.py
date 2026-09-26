# Hjaelpefil til trin 6: Send beskeder til en ekstern sprogmodel (LLM),
# dvs. en AI-model, der kan formulere tekst. Filen bruges via ask_llm;
# den er ikke et selvstaendigt trin, der skal startes manuelt.
import requests
import os
from dotenv import load_dotenv

# Indlaes eventuelle indstillinger fra en lokal .env-fil.
# OPENROUTER_API_KEY skal findes der eller som miljoevariabel.
# API-noeglen giver adgang til tjenesten: Del den ikke, og gem den ikke i Git.
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
# OpenRouter formidler kaldet til den valgte modeludbyder over internettet.
url = "https://openrouter.ai/api/v1/chat/completions"
# Headers angiver adgangsnoeglen og at beskederne sendes i JSON-format.
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# messages er en liste af beskeder med rolle og tekst fra trin 6.
# model vaelger udbyderens model; navnet er ikke den lokale embedding-model.
# Tilgaengelighed, priser og begraensninger kan aendre sig, ogsaa for :free.
def ask_llm(messages, model="deepseek/deepseek-chat-v3-0324:free"):
    data = {
        "model": model,
        "messages": messages
    }
    # Send hele beskedlisten til tjenesten, og vent paa dens svar.
    # Dette kraever internet og kan koste penge afhaengigt af den valgte model.
    response = requests.post(url, headers=headers, json=data)
    # Stop med en fejl ved fx afvist adgang eller en fejl hos tjenesten.
    response.raise_for_status()
    # Svaret indeholder flere felter; vi returnerer kun teksten i det foerste svar.
    return response.json()["choices"][0]["message"]["content"]
