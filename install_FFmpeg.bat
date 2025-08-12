@echo off
chcp 65001 >nul
echo Установка RAG Chatbot
echo ================================

echo 1. Поиск Python 3...
set PYTHON_CMD=

REM Проверяем py (Python Launcher)
where py >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto found_python
)

REM Проверяем python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    goto found_python
)

REM Проверяем python с фильтром по версии 3
for /f "delims=" %%i in ('where python 2^>nul') do (
    "%%i" --version 2>&1 | findstr "Python 3" >nul
    if !errorlevel! equ 0 (
        set PYTHON_CMD="%%i"
        goto found_python
    )
)

echo ОШИБКА: Python 3 не найден в системе!
echo.
echo Возможные решения:
echo 1. Установите Python 3 с https://www.python.org/downloads/
echo    При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH"
echo 2. Или запустите этот файл из командной строки, где Python 3 доступен
echo.
pause
exit /b 1

:found_python
echo Найден Python 3: %PYTHON_CMD%
%PYTHON_CMD% --version

echo 2. Создание виртуального окружения...
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось создать виртуальное окружение
    pause
    exit /b 1
)

echo 3. Активация виртуального окружения...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось активировать виртуальное окружение
    pause
    exit /b 1
)

echo 4. Обновление pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось обновить pip
)

echo 5. Установка зависимостей...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось установить зависимости
    echo Это может быть связано с версиями пакетов или необходимостью компиляции.
    echo Убедитесь, что у вас установлен Visual Studio Build Tools или Visual Studio.
    pause
    exit /b 1
)

echo 6. Проверка PyTorch и CUDA...
python -c "import torch; print(f'PyTorch версия: {torch.__version__}'); print(f'Доступна CUDA: {torch.cuda.is_available()}'); print(f'Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>nul
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось проверить PyTorch/CUDA. Убедитесь, что PyTorch установлен правильно.
)

echo 7. Установка FFmpeg через imageio...
REM Примечание: imageio.plugins.ffmpeg.download() устарел. Лучше установить вручную.
python -c "import imageio.plugins.ffmpeg; imageio.plugins.ffmpeg.download()" 2>nul
if %errorlevel% equ 0 (
    echo УСПЕШНО: FFmpeg установлен через imageio ^(устаревший метод^)
) else (
    echo ПРЕДУПРЕЖДЕНИЕ: Автоматическая установка FFmpeg через imageio не удалась или не поддерживается.
)

echo 8. Проверка FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo УСПЕШНО: FFmpeg доступен в системе
) else (
    echo ПРЕДУПРЕЖДЕНИЕ: FFmpeg не найден в PATH
    echo Рекомендуется установить FFmpeg вручную:
    echo   - Скачайте с https://www.gyan.dev/ffmpeg/builds/
    echo   - Выберите "release essentials" сборку ^(например, ffmpeg-release-essentials.zip^)
    echo   - Распакуйте архив в папку, например, C:\ffmpeg
    echo   - Добавьте путь C:\ffmpeg\bin в переменные среды PATH
    echo   - Перезапустите командную строку или компьютер после изменения PATH
)

echo.
echo Установка завершена!
echo.
echo Для запуска приложения:
echo 1. Запустите run.bat
echo 2. Или вручную:
echo    call venv\Scripts\activate.bat
echo    python app.py
echo.
echo Если вы планируете использовать GPU (RTX 4060 Ti):
echo - Убедитесь, что у вас установлены драйверы NVIDIA.
echo - Убедитесь, что PyTorch использует CUDA (см. сообщение выше).
echo.

pause