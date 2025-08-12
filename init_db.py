
# init_db.py
from src.database import db_manager
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        print("Инициализация базы данных...")
        db_manager.initialize_database()
        print("✅ База данных инициализирована успешно.")
    except Exception as e:
        print(f"❌ Ошибка инициализации базы данных: {e}")
        logger.error(f"Ошибка инициализации базы данных: {e}", exc_info=True)