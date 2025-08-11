# src/export_handler.py
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json
import os
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

def export_chat_to_pdf(chat_history, model_name, filename=None):
    """Экспорт истории чата в PDF с поддержкой кириллицы"""
    try:
        if not filename:
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Создаем PDF документ
        doc = SimpleDocTemplate(filename, pagesize=A4, 
                               leftMargin=50, rightMargin=50, 
                               topMargin=50, bottomMargin=50)
        
        # Регистрируем шрифты DejaVu (гарантированная поддержка кириллицы)
        try:
            # Путь к шрифтам в проекте
            font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
            font_path = os.path.join(font_dir, 'DejaVuSans.ttf')
            font_bold_path = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')
            
            if os.path.exists(font_path) and os.path.exists(font_bold_path):
                pdfmetrics.registerFont(TTFont('DejaVu', font_path))
                pdfmetrics.registerFont(TTFont('DejaVu-Bold', font_bold_path))
                font_name = 'DejaVu'
                font_bold_name = 'DejaVu-Bold'
                logger.info("Используются шрифты DejaVu")
            else:
                # fallback на системные шрифты
                raise FileNotFoundError("Шрифты DejaVu не найдены в проекте")
        except Exception as e:
            logger.warning(f"Не удалось загрузить шрифты DejaVu: {e}")
            # Пробуем системные шрифты
            try:
                calibri_path = os.path.join("C:", "Windows", "Fonts", "calibri.ttf")
                calibri_bold_path = os.path.join("C:", "Windows", "Fonts", "calibrib.ttf")
                
                if os.path.exists(calibri_path) and os.path.exists(calibri_bold_path):
                    pdfmetrics.registerFont(TTFont('Calibri', calibri_path))
                    pdfmetrics.registerFont(TTFont('Calibri-Bold', calibri_bold_path))
                    font_name = 'Calibri'
                    font_bold_name = 'Calibri-Bold'
                else:
                    # Если Calibri нет, пробуем Arial
                    arial_path = os.path.join("C:", "Windows", "Fonts", "arial.ttf")
                    arial_bold_path = os.path.join("C:", "Windows", "Fonts", "arialbd.ttf")
                    
                    if os.path.exists(arial_path) and os.path.exists(arial_bold_path):
                        pdfmetrics.registerFont(TTFont('Arial', arial_path))
                        pdfmetrics.registerFont(TTFont('Arial-Bold', arial_bold_path))
                        font_name = 'Arial'
                        font_bold_name = 'Arial-Bold'
                    else:
                        # Если системных шрифтов нет, используем встроенные (с ограничениями)
                        font_name = 'Helvetica'
                        font_bold_name = 'Helvetica-Bold'
                        logger.warning("Системные шрифты не найдены. Используются встроенные шрифты (кириллица может отображаться некорректно).")
            except Exception as e2:
                logger.warning(f"Не удалось загрузить системные шрифты: {e2}")
                font_name = 'Helvetica'
                font_bold_name = 'Helvetica-Bold'
        
        # Создаем стили с поддержкой кириллицы
        title_style = ParagraphStyle(
            'CustomTitle',
            fontName=font_bold_name,
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Центрирование
            textColor=colors.black
        )
        
        metadata_style = ParagraphStyle(
            'Metadata',
            fontName=font_name,
            fontSize=10,
            spaceAfter=20,
            textColor=colors.gray
        )
        
        user_label_style = ParagraphStyle(
            'UserLabel',
            fontName=font_bold_name,
            fontSize=12,
            spaceAfter=6,
            textColor=colors.blue
        )
        
        assistant_label_style = ParagraphStyle(
            'AssistantLabel',
            fontName=font_bold_name,
            fontSize=12,
            spaceAfter=6,
            textColor=colors.green
        )
        
        message_style = ParagraphStyle(
            'Message',
            fontName=font_name,
            fontSize=11,
            spaceAfter=15,
            textColor=colors.black
        )
        
        # Создаем содержимое
        story = []
        
        # Заголовок
        title = Paragraph("История чата RAG Chatbot", title_style)
        story.append(title)
        
        # Метаданные
        metadata_text = (
            f"<b>Модель:</b> {model_name or 'Не выбрана'}<br/>"
            f"<b>Дата экспорта:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        metadata = Paragraph(metadata_text, metadata_style)
        story.append(metadata)
        story.append(Spacer(1, 20))
        
        # Сообщения чата
        for user_msg, ai_msg in chat_history:
            # Сообщение пользователя
            user_label = Paragraph("Пользователь:", user_label_style)
            story.append(user_label)
            user_message = Paragraph(user_msg.replace('\n', '<br/>'), message_style)
            story.append(user_message)
            
            # Ответ ассистента
            assistant_label = Paragraph("Ассистент:", assistant_label_style)
            story.append(assistant_label)
            assistant_message = Paragraph(ai_msg.replace('\n', '<br/>'), message_style)
            story.append(assistant_message)
            
            # Разделитель между сообщениями
            story.append(Spacer(1, 15))
        
        # Сохраняем PDF
        doc.build(story)
        logger.info(f"Чат экспортирован в PDF: {filename}")
        return f"✅ Чат экспортирован в {filename}"
    
    except Exception as e:
        error_msg = f"❌ Ошибка экспорта в PDF: {str(e)}"
        logger.error(error_msg)
        return error_msg

def export_chat_to_json(chat_history, model_name, filename=None):
    """Экспорт истории чата в JSON"""
    try:
        if not filename:
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "model": model_name or "Не выбрана",
            "timestamp": datetime.now().isoformat(),
            "chat_history": chat_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Чат экспортирован в JSON: {filename}")
        return f"✅ Чат экспортирован в {filename}"
    
    except Exception as e:
        error_msg = f"❌ Ошибка экспорта в JSON: {str(e)}"
        logger.error(error_msg)
        return error_msg