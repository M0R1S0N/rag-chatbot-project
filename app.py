# app.py
import gradio as gr
import os
import json
from datetime import datetime
from src.document_processor import load_multiple_documents, split_documents
from src.vector_store import create_vectorstore, save_vectorstore, load_vectorstore
from src.chat_chain import create_rag_chain, format_sources
from src.llm_handler import get_llm, get_available_models
from src.export_handler import export_chat_to_pdf, export_chat_to_json
from config.settings import settings
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
vectorstore = None
qa_chain = None
current_model = None
chat_history = []  # Для экспорта

def process_documents(files):
    """Обработка загруженных документов"""
    global vectorstore
    try:
        if not files:
            return "❌ Не выбраны файлы для загрузки!"
        
        # Получаем пути к файлам
        file_paths = [f.name for f in files]
        
        # Загружаем все документы
        documents = load_multiple_documents(file_paths)
        
        if not documents:
            return "❌ Не удалось загрузить ни один документ!"
        
        # Разделяем на чанки
        texts = split_documents(documents)
        
        # Создаем векторное хранилище
        vectorstore = create_vectorstore(texts)
        save_vectorstore(vectorstore)
        
        return f"✅ Обработано {len(files)} файлов. Всего документов: {len(documents)}, чанков: {len(texts)}"
    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}"
        logger.error(error_msg)
        return error_msg

def initialize_chat(model_name_key):
    """Инициализация чат-бота с выбранной моделью"""
    global vectorstore, qa_chain, current_model
    try:
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
    global qa_chain, chat_history
    if qa_chain is None:
        return "", history, "Сначала инициализируйте чат-бота!"
    
    try:
        # Сохраняем текущий диалог
        chat_history.append((message, ""))  # Добавляем вопрос пользователя
        
        # Формируем чат-историю для RAG цепочки
        chat_history_formatted = [(human, ai) for human, ai in chat_history]
        
        result = qa_chain({"question": message, "chat_history": chat_history_formatted})
        answer = result["answer"]
        
        # Обновляем историю ответом бота
        chat_history[-1] = (chat_history[-1][0], answer)  # Обновляем последний элемент
        
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
    
    with gr.Tab("Документы"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Загрузить документы (PDF/TXT/DOCX/HTML/MD)", 
                    file_count="multiple"
                )
                process_btn = gr.Button("Обработать документы")
                status1 = gr.Textbox(label="Статус")
                process_btn.click(process_documents, inputs=file_input, outputs=status1)
            
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=list(get_available_models().keys()),
                    value="Claude Sonnet 4",
                    label="Выберите модель LLM"
                )
                init_btn = gr.Button("Инициализировать чат-бота")
                status2 = gr.Textbox(label="Статус")
                init_btn.click(initialize_chat, inputs=model_dropdown, outputs=[model_dropdown, status2, status1])
    
    with gr.Tab("Чат"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Диалог", height=500)
                with gr.Row():
                    msg = gr.Textbox(
                        label="Введите ваш вопрос", 
                        placeholder="Задайте вопрос по документам...",
                        scale=8
                    )
                    clear_btn = gr.Button("Очистить", scale=1)
                
                # Кнопки экспорта с выбором директории
                with gr.Row():
                    export_dir = gr.Textbox(
                        label="Директория для экспорта (опционально)", 
                        placeholder="Оставьте пустым для сохранения в текущую папку",
                        scale=6
                    )
                    export_json_btn = gr.Button("Экспорт в JSON", scale=2)
                    export_pdf_btn = gr.Button("Экспорт в PDF", scale=2)
                    export_status = gr.Textbox(label="Статус экспорта")
                
            with gr.Column(scale=1):
                sources_output = gr.Markdown(label="Источники", height=500)
        
        msg.submit(chat, [msg, chatbot], [msg, chatbot, sources_output])
        clear_btn.click(clear_chat, None, chatbot)
        # Используем обертки с передачей директории
        export_json_btn.click(
            export_chat_json_wrapper, 
            inputs=export_dir,
            outputs=export_status
        )
        export_pdf_btn.click(
            export_chat_pdf_wrapper, 
            inputs=export_dir,
            outputs=export_status
        )

if __name__ == "__main__":
    demo.launch()