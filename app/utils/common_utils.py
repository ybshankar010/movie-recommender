import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

def setup_llm(model_name: str = "llama3.1:8b", temperature: float = 0.1):
    """
    Setup the LLM with the specified model name and temperature.
    
    Args:
        model_name (str): The name of the LLM model to use.
        temperature (float): The temperature setting for the LLM.
        
    Returns:
        llm instance with specified settings.
    """
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_OPENAI_API_BASE", "http://localhost:11434/v1"),
        api_key=os.getenv("OLLAMA_OPENAI_API_KEY", "ollama")
    )

    return llm