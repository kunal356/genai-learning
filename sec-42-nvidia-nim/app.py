from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('NVIDIA_API_KEY') 
client = OpenAI(
    base_url= "https://integrate.api.nvidia.com/v1",
    api_key= api_key
)

completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=[{"role":"user","content":"Give me information on Inferencing in Generative AI"}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")