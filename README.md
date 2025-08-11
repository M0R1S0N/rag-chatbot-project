# 🤖 RAG Chatbot с расширенными возможностями

Профессиональный чат-бот с Retrieval-Augmented Generation (RAG), поддерживающий работу с различными LLM через OpenRouter API.

## 🌟 Возможности

- **Множественная загрузка документов** (PDF, TXT, DOCX, HTML, MD)
- **Выбор различных моделей LLM** через OpenRouter
- **Экспорт диалогов** в PDF и JSON
- **Работа с контекстом** диалога
- **Отображение источников** ответов
- **Поддержка кириллицы** в экспорте PDF

## 🚀 Технологии

- **LLM**: OpenRouter API (Claude, Gemini, Mistral, и др.)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **UI**: Gradio
- **Embeddings**: OpenAI Embeddings via OpenRouter
- **PDF Export**: ReportLab с поддержкой Unicode

## 📦 Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/ваш_логин/rag-chatbot.git
cd rag-chatbot

2. Создание виртуального окружения

# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Активация (macOS/Linux)
source .venv/bin/activate

3. Установка зависимостей

pip install -r requirements.txt

4. Настройка API ключа

# Скопируйте пример конфигурации
cp example.env .env

# Откройте файл и добавьте ваш API ключ OpenRouter
nano .env

▶️ Запуск: 

python app.py
Откройте в браузере: http://localhost:7860

📖 Использование
Загрузка документов:
    Перейдите на вкладку "Документы"
    Загрузите один или несколько файлов (PDF, TXT, DOCX, HTML, MD)
    Нажмите "Обработать документы"
    Выбор модели LLM:
    Выберите желаемую модель из списка
    Нажмите "Инициализировать чат-бота"
    Работа с чатом:
    Перейдите на вкладку "Чат"
    Задавайте вопросы по загруженным документам
    Просматривайте источники ответов
    Экспорт диалога:
    Используйте кнопки "Экспорт в JSON" или "Экспорт в PDF"
    Файлы сохраняются в корневой папке проекта

🔧 Поддерживаемые форматы документов
    PDF - Portable Document Format
    TXT - Текстовые файлы
    DOCX - Документы Microsoft Word
    HTML - Веб-страницы
    MD - Markdown файлы

🧪 Тестирование
# Запуск тестов (если есть)
    pytest tests/

📁 Структура проекта

rag-chatbot/
├── app.py                 # Главное приложение
├── requirements.txt       # Зависимости
├── .env                  # Конфигурация (не включается в репозиторий)
├── example.env           # Пример конфигурации
├── README.md             # Документация
├── .gitignore            # Игнорируемые файлы
├── config/               # Конфигурационные файлы
│   └── settings.py       # Настройки проекта
├── src/                  # Исходный код
│   ├── document_processor.py  # Обработка документов
│   ├── vector_store.py        # Работа с векторным хранилищем
│   ├── llm_handler.py         # Работа с LLM
│   ├── chat_chain.py          # Цепочка RAG
│   └── export_handler.py      # Экспорт в PDF/JSON
├── fonts/                # Шрифты для PDF (опционально)
│   ├── DejaVuSans.ttf
│   └── DejaVuSans-Bold.ttf
├── data/                 # Данные (создаются автоматически)
│   └── vectorstore/      # Векторное хранилище
└── logs/                 # Логи (создаются автоматически)


🔒 Безопасность
    API ключи хранятся в файле .env, который не включается в репозиторий
    Все секреты должны быть в .env файле
    Никогда не публикуйте файлы с API ключами

🤝 Вклад в проект
    Форкните репозиторий
    Создайте ветку для вашей функции (git checkout -b feature/AmazingFeature)
    Зафиксируйте изменения (git commit -m 'Add some AmazingFeature')
    Отправьте изменения в ветку (git push origin feature/AmazingFeature)
    Откройте Pull Request

📄 Лицензия
    Этот проект лицензирован по лицензии MIT - см. файл LICENSE для подробностей.

🙏 Благодарности
    OpenRouter - за предоставление доступа к различным LLM
    LangChain - за фреймворк
    Gradio - за простой веб-интерфейс
    ReportLab - за генерацию PDF
    
📞 Контакты
    Если у вас есть вопросы, создайте issue в этом репозитории.