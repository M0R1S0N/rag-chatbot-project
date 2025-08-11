import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader,  # Для DOCX файлов
    UnstructuredHTMLLoader,  # Для HTML файлов
    UnstructuredMarkdownLoader  # Для Markdown файлов
)
from typing import List
import logging

logger = logging.getLogger(__name__)

def load_document(file_path: str) -> List:
    """Загрузка документа различных форматов"""
    try:
        # Нормализуем путь
        file_path = os.path.normpath(file_path)
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        logger.info(f"Попытка загрузить файл: {file_path}")
        
        # Определяем тип файла и используем соответствующий лоадер
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".html") or file_path.endswith(".htm"):
            loader = UnstructuredHTMLLoader(file_path)
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            # fallback на TextLoader для неизвестных форматов
            logger.warning(f"Неизвестный формат файла: {file_path}. Используется TextLoader.")
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()    
        # Добавляем метаданные к каждому документу
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(file_path)
            doc.metadata["file_type"] = file_path.split('.')[-1].upper()
            logger.info(f"Загружено {len(documents)} документов из {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Ошибка загрузки документа {file_path}: {e}")
        raise

def load_multiple_documents(file_paths: List[str]) -> List:
    """Загрузка нескольких документов"""
    all_documents = []
    for file_path in file_paths:
        try:
            logger.info(f"Загрузка документа: {file_path}")
            # Проверяем, является ли файл медиа (уже обработанным текстом)
            if isinstance(file_path, tuple) and len(file_path) == 2:
                # Это кортеж (текст, имя_файла) от медиа обработки
                text, source_name = file_path
                doc = create_document_from_text(text, source_name)
                all_documents.append(doc)
                logger.info(f"Добавлен документ из медиа: {source_name}")
            else:
                # Обычный файловый путь
                if os.path.exists(file_path):
                    documents = load_document(file_path)
                    all_documents.extend(documents)
                    logger.info(f"Загружено {len(documents)} документов из {file_path}")
                else:
                    logger.error(f"Файл не найден: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {e}", exc_info=True)
            # Продолжаем загрузку остальных файлов
            continue
    logger.info(f"Всего загружено документов: {len(all_documents)}")
    return all_documents

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Разделение документов на чанки"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Разделено на {len(texts)} чанков")
        return texts
    except Exception as e:
        logger.error(f"Ошибка разделения документов: {e}")
        raise

def create_document_from_text(text: str, source_name: str) -> Document:
    """Создает документ из текста с метаданными"""
    return Document(
        page_content=text,
        metadata={
            "source_file": source_name,
            "file_type": "MEDIA"
        }
    )