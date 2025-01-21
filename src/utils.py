import os
from openai import OpenAI
from typing import List, Dict, Any


def chat_completion_without_stream(
    messages: List[Dict[str, str]],
    history: List[str] = [],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: str = None
) -> str:
    """
    Get a non-streaming completion from OpenAI's API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: OpenAI model to use
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        str: The model's response text
    """
    # Convert history into system message if there's history
    if history:
        history_message = {
            "role": "system",
            "content": "You are a helpful assistant. Here is the previous conversation history:\n" + "\n".join(history)
        }
        
    messages = [history_message] + messages
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat_completion_without_stream([{"role": "user", "content": "Hello, how are you?"}], history=["Hello, how are you?"], model="gpt-4o-mini"))