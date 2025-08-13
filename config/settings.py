import os
from dotenv import load_dotenv
import sys

# Пытаемся загрузить .env файл, если он существует
load_dotenv()

class Settings:
    # API ключ OpenRouter (обязательный параметр)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Доступные модели LLM
    AVAILABLE_MODELS = {
        "OpenAI: GPT-4.1": "openai/gpt-4.1",
        "Claude Sonnet 4": "anthropic/claude-sonnet-4",
        "Google: Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
        "Google: Gemini 2.5 Flash": "google/gemini-2.5-flash",
        "DeepSeek: DeepSeek V3 0324": "deepseek/deepseek-chat-v3-0324",
        "Anthropic: Claude 3.7 Sonnet": "anthropic/claude-3.7-sonnet",
        "Qwen: Qwen3 30B A3B": "qwen/qwen3-30b-a3b"
    }
    
    DEFAULT_MODEL = "anthropic/claude-sonnet-4"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    VECTOR_STORE_PATH = "data/vectorstore/faiss_index"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 2000
    
    def validate(self):
        """Проверка обязательных настроек"""
        if not self.OPENROUTER_API_KEY:
            print("⚠️  ВНИМАНИЕ: OPENROUTER_API_KEY не установлен!")
            print("Пожалуйста, создайте файл .env и добавьте в него ваш API ключ")
            print("Пример: cp example.env .env && nano .env")
            return False
        return True

settings = Settings()

# Проверяем настройки при импорте
if not settings.validate():
    print("Некоторые функции могут работать некорректно без API ключа")