from langchain_community.vectorstores import FAISS
from src.embeddings_handler import get_embeddings
from config.settings import settings
import os
import logging

logger = logging.getLogger(__name__)

def create_vectorstore(documents):
    """Создание векторного хранилища"""
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info("Векторное хранилище создано")
        return vectorstore
    except Exception as e:
        logger.error(f"Ошибка создания векторного хранилища: {e}")
        raise

def save_vectorstore(vectorstore, path: str = None):
    """Сохранение векторного хранилища"""
    if path is None:
        path = settings.VECTOR_STORE_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vectorstore.save_local(path)
        logger.info(f"Векторное хранилище сохранено в {path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения векторного хранилища: {e}")
        raise

def load_vectorstore(path: str = None):
    """Загрузка векторного хранилища"""
    if path is None:
        path = settings.VECTOR_STORE_PATH
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Векторное хранилище загружено из {path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Ошибка загрузки векторного хранилища: {e}")
        raise