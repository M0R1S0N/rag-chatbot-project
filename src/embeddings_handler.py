# src/embeddings_handler.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # Используется устаревший класс, но пусть пока работает
from config.settings import settings
import logging
import torch

logger = logging.getLogger(__name__)

def get_embeddings():
    """Получение embeddings модели с fallback на локальные"""
    try:
        # Сначала пробуем OpenAI embeddings через OpenRouter
        # Исправлено: убран лишний пробел в URL
        embeddings = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1", # Убран лишний пробел
            api_key=settings.OPENROUTER_API_KEY,
            model="text-embedding-ada-002"
        )
        # Проверяем, работают ли embeddings
        test_embedding = embeddings.embed_query("test")
        # Исправлено: Проверка должна быть более точной. API обычно возвращает объект, а не просто список.
        # Но если ваша логика работает, оставим. Главное, чтобы исключение ловилось.
        if hasattr(test_embedding, '__len__') and len(test_embedding) > 0: # Более общая проверка
             logger.info("Используются OpenAI embeddings через OpenRouter")
             return embeddings
        else:
            raise ValueError("Неправильный формат ответа от embeddings API")
    except Exception as e:
        logger.warning(f"Не удалось использовать OpenAI embeddings: {e}")
        # fallback на локальные embeddings
        try:
            # Определяем устройство для PyTorch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                logger.info(f"Embeddings: Используется устройство: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Embeddings: CUDA не доступна, используется CPU")

            # Передаем устройство в HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device} # <-- Добавлено
            )
            logger.info("Используются локальные HuggingFace embeddings (fallback)")
            return embeddings
        except Exception as e2:
            logger.error(f"Не удалось инициализировать локальные embeddings: {e2}")
            raise
