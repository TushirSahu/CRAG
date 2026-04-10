import google.genai as genai
from dotenv import load_dotenv
load_dotenv()
client = genai.Client()
for m in client.models.list():
    if "generateContent" in m.supported_actions:
        try:
            print(f"Testing {m.name}")
            response = client.models.generate_content(model=m.name, contents="Hi")
            print(f"{m.name} success!")
            break
        except Exception as e:
            # print(e)
            pass
