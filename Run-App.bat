@echo on
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title LectureTranscriptor Launcher

REM 1) В папку проекта
cd /d "%~dp0" || (echo [!] Не удалось перейти в каталог запуска & pause & exit /b 1)

echo ================== LAUNCHER START ==================
echo Папка проекта: %CD%

REM 2) Поиск Python
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE ( where python >nul 2>nul && set "PYEXE=python" )
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"

if not defined PYEXE (
  echo [!] Python 3.12 не найден. Установи Python 3.12 и запусти снова.
  pause & exit /b 1
)
echo Найден Python: %PYEXE%

REM 3) Создание/проверка venv
if not exist "env\Scripts\python.exe" (
  echo [*] Создаю виртуальное окружение env ...
  if /I "%PYEXE%"=="py" (
    py -3.12 -m venv env
  ) else (
    "%PYEXE%" -m venv env
  )
  if errorlevel 1 (
    echo [!] Не удалось создать venv. Проверь права/антивирус.
    pause & exit /b 1
  )
) else (
  echo [*] Обнаружено виртуальное окружение env
)

call "env\Scripts\activate.bat" || (echo [!] Не удалось активировать env & pause & exit /b 1)

REM 4) Установка зависимостей
set "REQ_LOG=%CD%\_setup_log.txt"
echo [*] Обновляю pip... (лог: %REQ_LOG%)
python -m pip --disable-pip-version-check -q install --upgrade pip >> "%REQ_LOG%" 2>&1

if exist requirements.txt (
  echo [*] Устанавливаю зависимости из requirements.txt ...
  python -m pip -q install -r requirements.txt >> "%REQ_LOG%" 2>&1
) else (
  echo [*] Проверяю/ставлю пакеты: streamlit faster-whisper pydub numpy ...
  python -m pip -q install streamlit faster-whisper pydub numpy >> "%REQ_LOG%" 2>&1
)
if errorlevel 1 (
  echo [!] Ошибка установки зависимостей. См. %REQ_LOG%
  notepad "%REQ_LOG%"
  pause & exit /b 1
)

REM 5) Запуск приложения
echo [*] Запускаю Streamlit...
streamlit run LectureTranscriptor.py
set "RC=%ERRORLEVEL%"
echo [*] Streamlit завершился с кодом %RC%.
echo =================== LAUNCHER END ====================
pause
exit /b %RC%
