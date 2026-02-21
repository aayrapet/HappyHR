import os
from dotenv import load_dotenv
from google import genai

# Charger le fichier .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your-gemini-api-key":
    print("❌ Erreur : La clé GEMINI_API_KEY n'est pas ou mal configurée dans le fichier .env.")
    exit(1)

print(f"✅ Clé lue depuis le .env (commence par : {api_key[:10]}...)")
print("🔄 Envoi d'une requête de test à l'API Google Gemini...")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Dis bonjour en français très brièvement."
    )
    print("\n🎉 SUCCÈS ! La clé est valide et fonctionne parfaitement.")
    print(f"🤖 Gemini a répondu : {response.text.strip()}")
except Exception as e:
    print("\n❌ ÉCHEC : L'API Google a refusé la requête.")
    print(f"Détails de l'erreur : {str(e)}")
