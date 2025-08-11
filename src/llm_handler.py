# src/llm_handler.py
from langchain_openai import ChatOpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

def get_llm(model_name: str = None):
    """Получение LLM модели через OpenRouter"""
    if model_name is None:
        model_name = settings.DEFAULT_MODEL

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
        model=model_name,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS
    )

    logger.info(f"LLM модель {model_name} инициализирована")
    return llm

def get_available_models():
    """Получение списка доступных моделей"""
    return settings.AVAILABLE_MODELS