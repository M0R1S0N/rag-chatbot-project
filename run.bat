@echo off
chcp 65001 >nul
title RAG Chatbot
echo Запуск RAG Chatbot...

if not exist venv (
    echo ОШИБКА: Виртуальное окружение 'venv' не найдено!
    echo Пожалуйста, сначала запустите install_FFmpeg.bat для создания окружения и установки зависимостей.
    pause
    exit /b 1
)

echo Активация виртуального окружения...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось активировать виртуальное окружение. Файл activate.bat не найден или поврежден.
    pause
    exit /b 1
)

echo Запуск приложения...
python app.py

if %errorlevel% neq 0 (
    echo.
    echo ОШИБКА: Приложение завершилось с кодом ошибки %errorlevel%.
    echo Проверьте логи выше для получения дополнительной информации.
)

echo.
echo Приложение завершено.
pause