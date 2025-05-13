from together import Together

client = Together()

response = client.chat.completions.create(
    model="nikilrav/Mixtral-8x7B-Instruct-v0.1-self_taught_eval_ft_v3_sanity_iter1-223a1d08-a5fdb2b8",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}]
)

print(response.choices[0].message.content)