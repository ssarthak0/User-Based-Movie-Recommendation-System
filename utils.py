import openai

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="Your groq api key",
)

def generate_summary(movie_titles):
    prompt = f"Suggest these movies in a friendly way: {', '.join(movie_titles)}."

    response = client.chat.completions.create(
        model="llama3-70b-8192",  # You can also use mixtral if you want
        messages=[
            {"role": "system", "content": "You are a movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    message = response.choices[0].message.content
    return message