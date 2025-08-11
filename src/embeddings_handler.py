from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

def get_embeddings():
    """Получение embeddings модели с fallback на локальные"""
    try:
        # Сначала пробуем OpenAI embeddings через OpenRouter
        embeddings = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            model="text-embedding-ada-002"
        )
        # Проверяем, работают ли embeddings
        test_embedding = embeddings.embed_query("test")
        if isinstance(test_embedding, list) and len(test_embedding) > 0:
            logger.info("Используются OpenAI embeddings через OpenRouter")
            return embeddings
        else:
            raise ValueError("Неправильный формат ответа от embeddings API")
    except Exception as e:
        logger.warning(f"Не удалось использовать OpenAI embeddings: {e}")
        # fallback на локальные embeddings
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Используются локальные HuggingFace embeddings (fallback)")
            return embeddings
        except Exception as e2:
            logger.error(f"Не удалось инициализировать локальные embeddings: {e2}")
            raise