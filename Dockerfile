# Используем базовый образ Python
FROM python:3.10-slim

# Установка системных зависимостей, включая ffmpeg
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Создание необходимых директорий
RUN mkdir -p data/vectorstore fonts

# Открытие порта для Gradio
EXPOSE 7860

# Команда для запуска приложения
CMD ["python", "app.py"]