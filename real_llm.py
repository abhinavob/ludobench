from openai import OpenAI
import time
client = OpenAI(
    base_url="https://openrouter.ai/api/v1"
)

# MODEL = "qwen/qwen-2.5-7b-instruct"  
# MODEL = "openai/gpt-oss-20b:free"
# MODEL = "openai/gpt-oss-120b:free"
# MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"
# MODEL = "qwen/qwen-plus"
# MODEL = "google/gemma-3-12b-it"
# MODEL = "meta-llama/llama-guard-4-12b"
MODEL = "meta-llama/llama-4-scout"
# MODEL = "openai/gpt-5-mini"
# MODEL = "openai/gpt-5-mini"
# MODEL = "anthropic/claude-3.5-haiku"
# MODEL = "google/gemma-3-27b-it:free"
# MODEL = "google/gemma-3-4b-it:free"
# MODEL = "anthropic/claude-sonnet-4.6"
# MODEL = "google/gemini-3.1-pro-preview"
# MODEL = "deepseek/deepseek-r1-0528:free"
# MODEL = "deepseek/deepseek-chat"
# MODEL = "openai/gpt-oss-20b"
# You can later switch to:
# MODEL= "deepseek/deepseek-r1-0528-qwen3-8b"
# MODEL="deepseek/deepseek-r1-0528:free"
# "openai/gpt-oss-20b"

def real_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Ludo player."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,   # IMPORTANT for reproducibility
        max_tokens=60,  # Allow short reasoning line
        stop=["\n"]      # Force single-line output
    )

    return response.choices[0].message.content.strip()




# def real_llm(prompt):

#     full_prompt = "You are an expert Ludo player.\n\n" + prompt

#     while True:
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": full_prompt}],
#                 temperature=0,
#                 max_tokens=10,
#                 stop=["\n"]
#             )

#             return response.choices[0].message.content.strip()

#         except Exception as e:
#             print("Rate limit hit, waiting 5 seconds...")
#             time.sleep(5)