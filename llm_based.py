from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Get an environment variable
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def generate_story_outline(text1, text2, advanced_model):
    messages=[
            {"role": "system", "content": "You are a creative assistant and author."},
            {"role": "user", "content": f"Can you come up with a short story that combines the texts \"{text1}\" with \"{text2}\"? Please first provide an outline by splitting the story into 5 sections. This outline should include the characters, setting, and the inciting incident. The first section should be exposition with an inciting incident, then the second section should contain rising action, the third should be the climax, then the falling action, and the resolutions. The rising action section should be the most detailed, while the other sections can be more concise."}
        ]
    response = client.chat.completions.create(
        model="gpt-4" if advanced_model else "gpt-3.5-turbo",
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    messages.append({"role": "system", "content": response.choices[0].message.content})
    return messages

def write_next_section(previous_messages, next_section_prompt, advanced_model):
    previous_messages.append({"role": "user", "content": next_section_prompt})
    response = client.chat.completions.create(
        model="gpt-4" if advanced_model else "gpt-3.5-turbo",
        messages=previous_messages,
        max_tokens=8092 if advanced_model else 4096,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    previous_messages.append({"role": "system", "content": response.choices[0].message.content})
    return previous_messages


def main(text1, text2, advanced_model=False):
    messages_so_far = generate_story_outline(text1, text2, advanced_model)
    sections = ["Exposition with an Inciting Incident section, this should be around 500 words",
     "Rising action section, this should be around 700 words",
     "Climax section, this should be around 500 words",
     "Falling action section, this should be around 400 words",
     "Resolution section, this should be around 400 words"]
    
    fullText = ""
    
    for section in sections:
        prompt = f"Write the next section of the story: {section}"
        messages_so_far = write_next_section(messages_so_far, prompt, advanced_model)
        fullText += messages_so_far[-1]["content"] + "\n"


    return fullText

# Example usage
# text1 = "The Great Gatsby"
# text2 = "1984"
# print(main(text1, text2))
