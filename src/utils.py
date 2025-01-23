import os
from openai import OpenAI
from typing import List, Dict, Any, AsyncGenerator
import json


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

async def chat_completion_with_stream(
    messages: List[Dict[str, str]],
    documents: List[str] = None,
    api_key: str = None,
    model: str = "gpt-4o-mini"
) -> AsyncGenerator[str, None]:
    """
    Generate streaming chat completion using OpenAI's API
    
    Args:
        messages: List of message dictionaries
        documents: Optional list of context documents
        api_key: OpenAI API key
        model: Model to use for completion
        
    Yields:
        Chunks of the response as they become available
    """
    client = OpenAI(api_key=api_key)
        
    if documents:
        context = "\n\nContext:\n" + "\n".join(documents)
        messages[-1]["content"] += context
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk and hasattr(chunk.choices[0], 'delta'):
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield json.dumps({
                        "type": "content",
                        "content": delta.content
                    })
                    
    except Exception as e:
        yield json.dumps({
            "type": "error",
            "content": str(e)
        })
