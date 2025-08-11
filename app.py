# app.py
import torch
import gradio as gr
import os
import json
from datetime import datetime
from src.document_processor import load_multiple_documents, split_documents
from src.vector_store import create_vectorstore, save_vectorstore, load_vectorstore
from src.chat_chain import create_rag_chain, format_sources
from src.llm_handler import get_llm, get_available_models
from src.export_handler import export_chat_to_pdf, export_chat_to_json
from src.database import db_manager
from config.settings import settings
import logging
import tempfile
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip

# Проверка наличия FFmpeg
try:
    import subprocess
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    print("✅ FFmpeg найден в системе")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("⚠️  FFmpeg не найден в системе")
    print("Пожалуйста, установите FFmpeg вручную:")
    print("   - Скачайте с https://www.gyan.dev/ffmpeg/builds/")
    print("   - Выберите 'release essentials' сборку")
    print("   - Распакуйте в C:\\ffmpeg")
    print("   - Добавьте C:\\ffmpeg\\bin в переменные среды PATH")


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка доступности CUDA для PyTorch
if torch.cuda.is_available():
    logger.info(f"PyTorch: Используется устройство: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: Количество доступных GPU: {torch.cuda.device_count()}")
else:
    logger.info("PyTorch: CUDA не доступна, используется CPU")

# Глобальные переменные
vectorstore = None
qa_chain = None
current_model = None
chat_history = []  # Формат: [(role, content), ...]
current_user_id = None
current_session_id = None

def extract_audio_from_video(video_path):
    """Извлекает аудио из видео файла"""
    try:
        audio_path = video_path.replace('.mp4', '.mp3').replace('.mov', '.mp3')
        clip = VideoFileClip(video_path)
        # Убираем устаревшие параметры
        clip.audio.write_audiofile(audio_path, logger=None)
        return audio_path
    except Exception as e:
        logger.error(f"Ошибка при извлечении аудио: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """Преобразует аудио в текст с помощью Whisper"""
    try:
        # Определение устройства (GPU или CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Whisper: Используется устройство: {torch.cuda.get_device_name(0)}")
        else:
            print("Whisper: CUDA не доступна, используется CPU")

        model = whisper.load_model("base").to(device) # Перемещаем модель на устройство
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Ошибка при преобразовании аудио: {str(e)}")
        return None

def process_media_file(file_obj):
    """Обрабатывает медиа файл и возвращает текст"""
    temp_path = None
    audio_path = None
    try:
        logger.info(f"Начало обработки медиа файла: {getattr(file_obj, 'name', 'unknown')}")
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
            # Получаем содержимое файла
            if hasattr(file_obj, 'read'):
                logger.info("Чтение файла через read()")
                file_obj.seek(0)
                file_content = file_obj.read()
            else:
                # Для NamedString читаем файл напрямую
                if hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                    logger.info("Чтение файла напрямую по пути")
                    with open(file_obj.name, 'rb') as f:
                        file_content = f.read()
                else:
                    logger.info("Преобразование содержимого в строку")
                    file_content = str(file_obj).encode('utf-8')
            
            tmp_file.write(file_content)
            temp_path = tmp_file.name
            logger.info(f"Временный файл создан: {temp_path}")

        file_extension = os.path.splitext(file_obj.name)[1].lower()
        logger.info(f"Расширение файла: {file_extension}")
        
        if file_extension in ['.mp3', '.wav']:
            # Аудио файл
            logger.info("Обработка аудио файла")
            text = transcribe_audio(temp_path)
            if text:
                logger.info(f"Аудио успешно транскрибировано, длина текста: {len(text)}")
            else:
                logger.warning("Транскрибирование аудио не дало результата")
        elif file_extension in ['.mp4', '.mov']:
            # Видео файл
            logger.info("Обработка видео файла")
            audio_path = extract_audio_from_video(temp_path)
            if audio_path and os.path.exists(audio_path):
                logger.info(f"Аудио извлечено: {audio_path}")
                text = transcribe_audio(audio_path)
                if text:
                    logger.info(f"Видео успешно транскрибировано, длина текста: {len(text)}")
                else:
                    logger.warning("Транскрибирование видео не дало результата")
            else:
                logger.error("Не удалось извлечь аудио из видео")
                text = None
        else:
            logger.warning(f"Неподдерживаемый формат медиа файла: {file_extension}")
            text = None
            
        return text
    except Exception as e:
        logger.error(f"Ошибка при обработке медиа файла {getattr(file_obj, 'name', 'unknown')}: {str(e)}", exc_info=True)
        return None
    finally:
        # Очищаем временные файлы
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Удален временный файл: {temp_path}")
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info(f"Удален временный аудио файл: {audio_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить временные файлы: {e}")
    
def try_load_vectorstore():
    """Пробует загрузить векторное хранилище с диска при старте"""
    global vectorstore
    try:
        vectorstore = load_vectorstore()
        logger.info("Векторное хранилище успешно загружено с диска.")
        return True
    except Exception as e:
        logger.warning(f"Векторное хранилище не найдено или не удалось загрузить: {e}")
        return False

# Вызов функции при старте приложения
try_load_vectorstore()

def initialize_database():
    """Инициализация базы данных"""
    try:
        db_manager.initialize_database()
        return "✅ База данных инициализирована"
    except Exception as e:
        error_msg = f"❌ Ошибка инициализации БД: {str(e)}"
        logger.error(error_msg)
        return error_msg

def login_user(username):
    """Вход пользователя"""
    global current_user_id
    try:
        if not username:
            return "", "❌ Введите имя пользователя"
        
        user_id = db_manager.create_user(username)
        current_user_id = user_id
        return "", f"✅ Добро пожаловать, {username}!"
    except Exception as e:
        error_msg = f"❌ Ошибка входа: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

def create_new_session(session_name):
    """Создание новой сессии"""
    global current_session_id, chat_history
    try:
        if not current_user_id:
            return "❌ Сначала войдите в систему"
        
        if not session_name:
            session_name = f"Сессия {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session_id = db_manager.create_session(current_user_id, session_name)
        current_session_id = session_id
        chat_history = []  # Очищаем историю для новой сессии
        return f"✅ Создана сессия: {session_name}"
    except Exception as e:
        error_msg = f"❌ Ошибка создания сессии: {str(e)}"
        logger.error(error_msg)
        return error_msg

def load_user_sessions():
    """Загрузка сессий пользователя"""
    try:
        if not current_user_id:
            return []
        
        sessions = db_manager.get_user_sessions(current_user_id)
        # Возвращаем список кортежей (label, value) для Gradio Dropdown
        choices = [
            (f"{s['session_name']} ({s['updated_at'].strftime('%Y-%m-%d %H:%M')})", s['id'])
            for s in sessions
        ]
        return choices
    except Exception as e:
        logger.error(f"Ошибка загрузки сессий: {e}")
        return []

def load_session(session_id):
    """Загрузка выбранной сессии"""
    global current_session_id, chat_history
    try:
        if not session_id:
            return [], "❌ Выберите сессию"
        
        # Получаем сообщения из базы данных
        messages = db_manager.get_session_messages(session_id)
        chat_history = messages  # Сохраняем в формате [(role, content), ...]
        current_session_id = session_id

        # Формируем пары (user, assistant) для gr.Chatbot
        formatted_history = []
        i = 0
        while i < len(messages):
            if messages[i][0] == "user":
                user_msg = messages[i][1]
                # Ищем следующий ответ ассистента
                if i + 1 < len(messages) and messages[i+1][0] == "assistant":
                    assistant_msg = messages[i+1][1]
                    formatted_history.append((user_msg, assistant_msg))
                    i += 2
                else:
                    # Если нет ответа ассистента, добавляем пустой
                    formatted_history.append((user_msg, ""))
                    i += 1
            else:
                # Если первый или неожиданный ассистент — пропускаем
                i += 1

        logger.info(f"Загружена история диалога: {formatted_history}")
        return formatted_history, f"✅ Загружена сессия {session_id}"
    except Exception as e:
        error_msg = f"❌ Ошибка загрузки сессии: {str(e)}"
        logger.error(error_msg)
        return [], error_msg

def process_documents(files):
    """Обработка загруженных документов"""
    global vectorstore
    try:
        if not files:
            return "❌ Не выбраны файлы для загрузки!"
        
        logger.info(f"Получено файлов для обработки: {len(files)}")
        
        all_documents = []
        media_texts = []  # Для хранения текстов из медиа файлов
        
        for i, file_obj in enumerate(files):
            try:
                logger.info(f"Обработка файла {i+1}/{len(files)}: {file_obj.name}")
                file_extension = os.path.splitext(file_obj.name)[1].lower()
                logger.info(f"Расширение файла: {file_extension}")
                
                # Обработка текстовых документов
                if file_extension in ['.txt', '.pdf', '.docx', '.html', '.md']:
                    logger.info(f"Загрузка текстового документа: {file_obj.name}")
                    documents = load_multiple_documents([file_obj.name])
                    if documents:
                        logger.info(f"Загружено {len(documents)} документов из {file_obj.name}")
                        all_documents.extend(documents)
                    else:
                        logger.warning(f"Не удалось загрузить документы из {file_obj.name}")
                
                # Обработка медиа файлов
                elif file_extension in ['.mp3', '.wav', '.mp4', '.mov']:
                    logger.info(f"Обработка медиа файла: {file_obj.name}")
                    text = process_media_file(file_obj)
                    if text:
                        logger.info(f"Извлечено {len(text)} символов из {file_obj.name}")
                        # Создаем кортеж (текст, имя_файла) для последующей обработки
                        media_texts.append((text, file_obj.name))
                    else:
                        logger.warning(f"Не удалось извлечь текст из {file_obj.name}")
                else:
                    logger.warning(f"Неподдерживаемый формат файла: {file_extension}")
            
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_obj.name}: {str(e)}")
                continue
        
        logger.info(f"Всего текстовых документов: {len(all_documents)}")
        logger.info(f"Всего медиа текстов: {len(media_texts)}")
        
        # Загружаем текстовые документы
        if all_documents:
            texts = split_documents(all_documents)
            logger.info(f"Разделено текстовые документы на {len(texts)} чанков")
        else:
            texts = []
        
        # Обрабатываем медиа тексты
        for text, source_name in media_texts:
            try:
                from src.document_processor import create_document_from_text
                doc = create_document_from_text(text, source_name)
                doc_texts = split_documents([doc])
                texts.extend(doc_texts)
                logger.info(f"Обработан медиа текст из {source_name}, добавлено {len(doc_texts)} чанков")
            except Exception as e:
                logger.error(f"Ошибка при обработке медиа текста из {source_name}: {str(e)}")
        
        if not texts:
            logger.error("Не удалось обработать ни один файл - нет текстов для векторизации")
            return "❌ Не удалось обработать ни один файл!"
        
        # Создаем векторное хранилище
        logger.info("Создание векторного хранилища...")
        vectorstore = create_vectorstore(texts)
        save_vectorstore(vectorstore)
        logger.info("Векторное хранилище создано и сохранено")
        
        total_files = len([f for f in files if os.path.splitext(f.name)[1].lower() in ['.txt', '.pdf', '.docx', '.html', '.md']]) + len(media_texts)
        return f"✅ Обработано {total_files} файлов. Всего чанков: {len(texts)}"
    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}"
        logger.error(error_msg, exc_info=True)  # Добавляем трассировку стека
        return error_msg

def initialize_chat(model_name_key):
    """Инициализация чат-бота с выбранной моделью"""
    global vectorstore, qa_chain, current_model
    try:
        # Проверяем, есть ли векторное хранилище
        if vectorstore is None:
            # Пытаемся загрузить ещё раз
            if not try_load_vectorstore():
                return "", "Сначала обработайте документы!", ""
        
        # Если всё равно нет векторного хранилища
        if vectorstore is None:
            return "", "Сначала обработайте документы!", ""
        
        # Получаем полное имя модели
        available_models = get_available_models()
        model_name = available_models.get(model_name_key, settings.DEFAULT_MODEL)
        current_model = model_name_key
        
        llm = get_llm(model_name)
        qa_chain = create_rag_chain(vectorstore, llm)
        message = f"✅ Чат-бот готов к работе! Используется {model_name_key}"
        return "", message, ""
    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}"
        logger.error(error_msg)
        return "", error_msg, ""

def chat(message, history):
    """Функция чата с отображением источников"""
    global qa_chain, chat_history, current_session_id
    if qa_chain is None:
        return "", history, "Сначала инициализируйте чат-бота!"
    
    try:
        # Сохраняем текущий диалог в глобальной истории
        chat_history.append(("user", message))  # Добавляем вопрос пользователя
        
        # Формируем чат-историю для RAG цепочки
        # Преобразуем в формат [(user_message, assistant_message), ...]
        chat_history_pairs = []
        temp_history = chat_history.copy()
        i = 0
        while i < len(temp_history):
            if temp_history[i][0] == "user":
                user_msg = temp_history[i][1]
                if i + 1 < len(temp_history) and temp_history[i+1][0] == "assistant":
                    assistant_msg = temp_history[i+1][1]
                    chat_history_pairs.append((user_msg, assistant_msg))
                    i += 2
                else:
                    chat_history_pairs.append((user_msg, ""))
                    i += 1
            else:
                i += 1
        
        result = qa_chain({"question": message, "chat_history": chat_history_pairs})
        answer = result["answer"]
        
        # Обновляем историю ответом бота
        chat_history.append(("assistant", answer))
        
        # Сохраняем сообщения в базу данных
        if current_session_id:
            try:
                db_manager.save_message(current_session_id, "user", message)
                db_manager.save_message(current_session_id, "assistant", answer)
            except Exception as e:
                logger.error(f"Ошибка сохранения сообщений в БД: {e}")
        
        # Форматируем источники
        sources = format_sources(result["source_documents"])
        sources_text = "📚 **Источники:**\n\n"
        for i, source in enumerate(sources[:3], 1):  # Показываем первые 3 источника
            sources_text += f"**Источник {i}:**\n"
            sources_text += f"{source['content']}\n"
            if source['metadata']:
                sources_text += f"*Метаданные: {source['metadata']}*\n\n"
        
        return "", history + [(message, answer)], sources_text
    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}"
        return "", history, error_msg

def clear_chat():
    """Очистка истории чата"""
    global chat_history
    chat_history = []
    return []

# Функции экспорта с выбором директории
def export_chat_json_wrapper(export_dir=""):
    """Обертка для экспорта чата в JSON с выбором директории"""
    global chat_history, current_model
    try:
        # Создаем имя файла
        filename_base = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Если указана директория и она существует, сохраняем там
        if export_dir and os.path.exists(export_dir):
            filename = os.path.join(export_dir, f"{filename_base}.json")
        else:
            # Иначе сохраняем в текущую директорию
            filename = f"{filename_base}.json"
        
        result = export_chat_to_json(chat_history, current_model, filename)
        return result
    except Exception as e:
        error_msg = f"❌ Ошибка экспорта: {str(e)}"
        logger.error(error_msg)
        return error_msg

def export_chat_pdf_wrapper(export_dir=""):
    """Обертка для экспорта чата в PDF с выбором директории"""
    global chat_history, current_model
    try:
        # Создаем имя файла
        filename_base = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Если указана директория и она существует, сохраняем там
        if export_dir and os.path.exists(export_dir):
            filename = os.path.join(export_dir, f"{filename_base}.pdf")
        else:
            # Иначе сохраняем в текущую директорию
            filename = f"{filename_base}.pdf"
        
        result = export_chat_to_pdf(chat_history, current_model, filename)
        return result
    except Exception as e:
        error_msg = f"❌ Ошибка экспорта: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Интерфейс Gradio
with gr.Blocks(title="RAG Chatbot Advanced") as demo:
    gr.Markdown("# 🤖 RAG Chatbot с расширенными возможностями")
    gr.Markdown("Профессиональный чат-бот с Retrieval-Augmented Generation")
    
    # Объявляем компоненты заранее
    chatbot = gr.Chatbot(label="Диалог", height=500)
    sources_output = gr.Markdown(label="Источники", height=500)
    
    with gr.Tab("1. Авторизация"):
        username_input = gr.Textbox(label="Имя пользователя", placeholder="Введите ваше имя")
        login_btn = gr.Button("Войти")
        login_status = gr.Textbox(label="Статус", interactive=False)
        login_btn.click(login_user, inputs=username_input, outputs=[username_input, login_status])

    with gr.Tab("2. Сессии"):
        sessions_dropdown = gr.Dropdown(label="Выберите сессию", choices=[], interactive=True)
        refresh_sessions_btn = gr.Button("Обновить список")
        load_session_btn = gr.Button("Загрузить сессию")
        session_load_status = gr.Textbox(label="Статус загрузки", interactive=False)
        session_name_input = gr.Textbox(label="Название новой сессии", placeholder="Оставьте пустым для автоматического названия")
        create_session_btn = gr.Button("Создать новую сессию")
        session_status = gr.Textbox(label="Статус сессии", interactive=False)

        # Исправленный обработчик обновления списка сессий
        def refresh_sessions_wrapper():
            choices = load_user_sessions()
            return gr.update(choices=choices, value=None)
        
        refresh_sessions_btn.click(refresh_sessions_wrapper, outputs=sessions_dropdown)
        create_session_btn.click(create_new_session, inputs=session_name_input, outputs=session_status)
        load_session_btn.click(load_session, inputs=sessions_dropdown, outputs=[chatbot, session_load_status])

    with gr.Tab("3. Загрузка документов"):
        file_input = gr.File(
        label="Загрузить документы (PDF/TXT/DOCX/HTML/MD/MP3/WAV/MP4/MOV)", 
        file_count="multiple",
        file_types=[".txt", ".pdf", ".docx", ".html", ".md", ".mp3", ".wav", ".mp4", ".mov"]
        )
        process_btn = gr.Button("Обработать документы")
        status1 = gr.Textbox(label="Статус", interactive=False)
        process_btn.click(process_documents, inputs=file_input, outputs=status1)

    with gr.Tab("4. Инициализация модели"):
        model_dropdown = gr.Dropdown(
            choices=list(get_available_models().keys()),
            value="Claude Sonnet 4",
            label="Выберите модель LLM"
        )
        init_btn = gr.Button("Инициализировать чат-бота")
        status2 = gr.Textbox(label="Статус", interactive=False)
        init_btn.click(initialize_chat, inputs=model_dropdown, outputs=[model_dropdown, status2, status1])

    with gr.Tab("5. Чат"):
        # chatbot и sources_output уже объявлены выше
        msg = gr.Textbox(label="Введите ваш вопрос", placeholder="Задайте вопрос по документам...")
        clear_btn = gr.Button("Очистить")
        msg.submit(chat, [msg, chatbot], [msg, chatbot, sources_output])
        clear_btn.click(clear_chat, None, chatbot)

    with gr.Tab("6. Экспорт"):
        export_dir = gr.Textbox(label="Директория для экспорта (опционально)", placeholder="Оставьте пустым для сохранения в текущую папку")
        export_json_btn = gr.Button("Экспорт в JSON")
        export_pdf_btn = gr.Button("Экспорт в PDF")
        export_status = gr.Textbox(label="Статус экспорта", interactive=False)
        export_json_btn.click(export_chat_json_wrapper, inputs=export_dir, outputs=export_status)
        export_pdf_btn.click(export_chat_pdf_wrapper, inputs=export_dir, outputs=export_status)

if __name__ == "__main__":
    demo.launch()
