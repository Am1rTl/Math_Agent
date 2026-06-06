from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="kwaipilot/kat-coder-pro:free",
  messages=[
              {
                "role": "user",
                "content": "What is the meaning of life?"
              }
            ]
)
print(completion.choices[0].message.content)
