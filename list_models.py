from google import genai
from dotenv import load_dotenv
load_dotenv()
try:
    client = genai.Client()
    for m in client.models.list():
        if "generateContent" in m.supported_actions:
            print(m.name)
except Exception as e:
    print(e)
