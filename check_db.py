# check_db.py
import os
import sys
import logging
from src.database import db_manager

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database():
    """Проверка содержимого базы данных"""
    try:
        print("=== Проверка базы данных ===\n")
        
        # Проверяем пользователей
        print("1. Пользователи:")
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                conn.set_client_encoding('UTF8')
                cursor.execute("SET client_encoding = 'UTF8'")
                cursor.execute("SELECT id, username, created_at, last_active FROM users ORDER BY created_at")
                users = cursor.fetchall()
                if users:
                    for user in users:
                        # Очищаем данные от проблемных символов
                        user_id = user[0]
                        username = user[1].encode('utf-8', errors='ignore').decode('utf-8') if user[1] else ""
                        created_at = user[2]
                        last_active = user[3]
                        print(f"   ID: {user_id}, Имя: {username}, Создан: {created_at}, Активность: {last_active}")
                else:
                    print("   Нет пользователей")
        
        print("\n2. Сессии:")
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                conn.set_client_encoding('UTF8')
                cursor.execute("SET client_encoding = 'UTF8'")
                cursor.execute("""
                    SELECT cs.id, cs.session_name, u.username, cs.created_at, cs.updated_at 
                    FROM chat_sessions cs 
                    JOIN users u ON cs.user_id = u.id 
                    ORDER BY cs.updated_at DESC
                """)
                sessions = cursor.fetchall()
                if sessions:
                    for session in sessions:
                        session_id = session[0]
                        session_name = session[1].encode('utf-8', errors='ignore').decode('utf-8') if session[1] else ""
                        username = session[2].encode('utf-8', errors='ignore').decode('utf-8') if session[2] else ""
                        created_at = session[3]
                        updated_at = session[4]
                        print(f"   ID: {session_id}, Название: {session_name}, Пользователь: {username}")
                        print(f"      Создана: {created_at}, Обновлена: {updated_at}")
                else:
                    print("   Нет сессий")
        
        print("\n3. Сообщения:")
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                conn.set_client_encoding('UTF8')
                cursor.execute("SET client_encoding = 'UTF8'")
                cursor.execute("""
                    SELECT cm.id, cm.role, LENGTH(cm.content) as content_length, cm.created_at, cs.session_name
                    FROM chat_messages cm
                    JOIN chat_sessions cs ON cm.session_id = cs.id
                    ORDER BY cm.created_at
                """)
                messages = cursor.fetchall()
                if messages:
                    for msg in messages:
                        msg_id = msg[0]
                        role = msg[1].encode('utf-8', errors='ignore').decode('utf-8') if msg[1] else ""
                        content_length = msg[2]
                        created_at = msg[3]
                        session_name = msg[4].encode('utf-8', errors='ignore').decode('utf-8') if msg[4] else ""
                        print(f"   ID: {msg_id}, Роль: {role}, Длина: {content_length} символов")
                        print(f"      Сессия: {session_name}, Время: {created_at}")
                else:
                    print("   Нет сообщений")
        
        print("\n=== Статистика ===")
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                conn.set_client_encoding('UTF8')
                cursor.execute("SET client_encoding = 'UTF8'")
                cursor.execute("SELECT COUNT(*) FROM users")
                users_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chat_sessions")
                sessions_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chat_messages")
                messages_count = cursor.fetchone()[0]
                
                print(f"Пользователей: {users_count}")
                print(f"Сессий: {sessions_count}")
                print(f"Сообщений: {messages_count}")
                
    except Exception as e:
        print(f"❌ Ошибка проверки базы данных: {e}")
        logger.error(f"Ошибка проверки базы данных: {e}", exc_info=True)

if __name__ == "__main__":
    check_database()