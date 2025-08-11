# src/database.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'rag_chatbot'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'port': os.getenv('DB_PORT', '5432'),
        }
        print("Параметры подключения к БД:", self.connection_params)
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения с БД"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.set_client_encoding('UTF8')
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка работы с базой данных: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self):
        """Инициализация базы данных (создание таблиц)"""
        try:
            # Читаем SQL файл для инициализации
            sql_file_path = os.path.join(os.path.dirname(__file__), '..', 'migrations', 'init.sql')
            if os.path.exists(sql_file_path):
                with open(sql_file_path, 'r', encoding='utf-8') as f:
                    sql_script = f.read()
                
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Устанавливаем кодировку перед выполнением
                        conn.set_client_encoding('UTF8')
                        cursor.execute("SET client_encoding = 'UTF8'")
                        cursor.execute(sql_script)
                        conn.commit()
                logger.info("База данных инициализирована успешно")
            else:
                logger.warning("Файл инициализации базы данных не найден")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            raise
    
    def create_user(self, username: str) -> int:
        """Создание нового пользователя"""
        try:
            # Очищаем имя пользователя от проблемных символов
            cleaned_username = username.encode('utf-8', errors='ignore').decode('utf-8')
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute(
                        "INSERT INTO users (username) VALUES (%s) ON CONFLICT (username) DO UPDATE SET last_active = CURRENT_TIMESTAMP RETURNING id",
                        (cleaned_username,)
                    )
                    user_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(f"Пользователь {cleaned_username} создан/обновлен с ID {user_id}")
                    return user_id
        except Exception as e:
            logger.error(f"Ошибка создания пользователя {username}: {e}")
            raise
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Получение ID пользователя по имени"""
        try:
            # Очищаем имя пользователя
            cleaned_username = username.encode('utf-8', errors='ignore').decode('utf-8')
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute("SELECT id FROM users WHERE username = %s", (cleaned_username,))
                    result = cursor.fetchone()
                    return result[0] if result else None
        except Exception as e:
            logger.error(f"Ошибка получения ID пользователя {username}: {e}")
            return None
    
    def create_session(self, user_id: int, session_name: str) -> int:
        """Создание новой сессии диалога"""
        try:
            # Очищаем название сессии
            cleaned_session_name = session_name.encode('utf-8', errors='ignore').decode('utf-8')
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute(
                        "INSERT INTO chat_sessions (user_id, session_name) VALUES (%s, %s) RETURNING id",
                        (user_id, cleaned_session_name)
                    )
                    session_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(f"Сессия '{cleaned_session_name}' создана с ID {session_id}")
                    return session_id
        except Exception as e:
            logger.error(f"Ошибка создания сессии для пользователя {user_id}: {e}")
            raise
    
    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """Получение всех сессий пользователя"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute(
                        "SELECT id, session_name, created_at, updated_at FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC",
                        (user_id,)
                    )
                    results = cursor.fetchall()
                    # Очищаем названия сессий
                    cleaned_results = []
                    for row in results:
                        cleaned_row = dict(row)
                        if cleaned_row['session_name']:
                            cleaned_row['session_name'] = cleaned_row['session_name'].encode('utf-8', errors='ignore').decode('utf-8')
                        cleaned_results.append(cleaned_row)
                    return cleaned_results
        except Exception as e:
            logger.error(f"Ошибка получения сессий пользователя {user_id}: {e}")
            return []
    
    def save_message(self, session_id: int, role: str, content: str):
        """Сохранение сообщения в сессии"""
        try:
            # Очищаем содержимое от проблемных символов
            cleaned_content = content.encode('utf-8', errors='ignore').decode('utf-8')
            cleaned_role = role.encode('utf-8', errors='ignore').decode('utf-8')
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute(
                        "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
                        (session_id, cleaned_role, cleaned_content)
                    )
                    conn.commit()
                    # Обновляем время сессии (триггер сработает автоматически)
                    cursor.execute(
                        "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (session_id,)
                    )
                    conn.commit()
                    logger.info(f"Сообщение сохранено в сессии {session_id}")
        except Exception as e:
            logger.error(f"Ошибка сохранения сообщения в сессии {session_id}: {e}")
            raise
    
    def get_session_messages(self, session_id: int) -> List[Tuple[str, str]]:
        """Получение всех сообщений сессии"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute(
                        "SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
                        (session_id,)
                    )
                    results = cursor.fetchall()
                    # Очищаем содержимое от проблемных символов
                    cleaned_results = []
                    for role, content in results:
                        try:
                            cleaned_role = role.encode('utf-8', errors='ignore').decode('utf-8') if role else role
                            cleaned_content = content.encode('utf-8', errors='ignore').decode('utf-8') if content else content
                            cleaned_results.append((cleaned_role, cleaned_content))
                        except:
                            # Если не можем очистить, сохраняем как есть
                            cleaned_results.append((role, content))
                    return cleaned_results
        except Exception as e:
            logger.error(f"Ошибка получения сообщений сессии {session_id}: {e}")
            return []
    
    def delete_session(self, session_id: int):
        """Удаление сессии и всех сообщений"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    conn.set_client_encoding('UTF8')
                    cursor.execute("SET client_encoding = 'UTF8'")
                    cursor.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                    conn.commit()
                    logger.info(f"Сессия {session_id} удалена")
        except Exception as e:
            logger.error(f"Ошибка удаления сессии {session_id}: {e}")
            raise

# Глобальный экземпляр менеджера базы данных
db_manager = DatabaseManager()