import os
from openai import OpenAI
from typing import List, Dict, Any


def chat_completion_without_stream(messages, model="gpt-4o-mini", api_key=None, documents=None):
    """
    Non-streaming version of chat completion
    """
    if documents:
        # Convert messages list to include document context
        history_message = {"role": "system", "content": f"Here are the documents:\n{'\n'.join(documents)}"} 
        messages = messages + [history_message]  # Correctly concatenate lists
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        return None
