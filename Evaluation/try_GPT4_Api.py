from openai import OpenAI

## Set the API key
client = OpenAI(api_key="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")

MODEL="gpt-4o"

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are an evaluation system and rate outputs by another model."},
    # {"role": "user", "content": f"An LLM was queried with ### {} ### and answered with ### {} ### The actual answer is ### {} ###. Is the answer provided by the"
    #                            "LLM corrcet? Answer with 1, if yes, or 0, if no."}
  ]
)
print("Assistant: " + completion.choices[0].message.content)