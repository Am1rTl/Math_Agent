from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_body={},
  model="mistralai/mistral-small-3.2-24b-instruct:free",
  messages=[
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "What is in this  and image?"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                  }
                ]
              }
            ]
)
print(completion.choices[0].message.content)
