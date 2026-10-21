from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
client = Groq(api_key=key)
ids = sorted(m.id for m in client.models.list().data)
for model_id in ids:
    print(model_id)
