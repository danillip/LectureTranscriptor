import os
import sys
import re
import math
import time
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import streamlit as st

# ---------- ПОРТАТИВНЫЕ ПУТИ/ПЕРЕМЕННЫЕ (обязаны быть САМЫМИ ВЕРХОМ) ----------
BASE_DIR: Path = Path(__file__).parent.resolve()

# Кэши под локальную папку, чтобы всё было «портативно»
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / "_hf_cache"))
os.environ.setdefault("HF_HOME",              str(BASE_DIR / "_hf_cache"))
os.environ.setdefault("CT2_DIR",              str(BASE_DIR / "_ct2_cache"))
os.environ.setdefault("XDG_CACHE_HOME",       str(BASE_DIR / "_hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Поддержка длинных путей Windows и корректного поиска DLL
if os.name == "nt":
    import ctypes
    try:
        ctypes.windll.kernel32.SetDefaultDllDirectories(0x1000)  # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    except Exception:
        pass

# --- Глушим предупреждение ctranslate2 про pkg_resources (не влияет на работу) ---
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"ctranslate2(\.|$)"
)

# ====================== Автоподключение CUDA/cuDNN DLL на Windows ======================

def _register_dll_dir(p: Path) -> bool:
    """Добавляет каталог p в поиск DLL (os.add_dll_directory + PATH)."""
    if not p or not p.exists():
        return False
    try:
        os.add_dll_directory(str(p))  # Py 3.8+
    except Exception:
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    return True

def find_and_register_cudnn() -> List[Path]:
    """
    Ищет файлы cudnn*64_9.dll в типовых местах и регистрирует их каталоги.
    Возвращает список добавленных путей. Ничего не делает на non-Windows.
    """
    if os.name != "nt":
        return []

    added: List[Path] = []
    dll_candidates = {
        "cudnn_ops64_9.dll",
        "cudnn_cnn_infer64_9.dll",
        "cudnn_cnn_train64_9.dll",
        "cudnn64_9.dll",
    }

    # 1) Корни для поиска
    roots: List[Path] = []

    # Переменные окружения CUDA (CUDA_PATH, CUDA_PATH_V*)
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH") and v:
            roots.append(Path(v))

    # Program Files стандартный путь CUDA Toolkit
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    roots.append(Path(pf) / "NVIDIA GPU Computing Toolkit" / "CUDA")

    # Иногда ставят отдельно cuDNN
    roots.append(Path(pf) / "NVIDIA" / "CUDNN")
    roots.append(Path(pf) / "NVIDIA" / "cuDNN")
    roots.append(Path(pf) / "NVIDIA Corporation")

    # 2) Кандидаты bin-папок
    bin_dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        # .\v12.6\bin, v12.5\bin, ...
        for sub in root.glob("v*"):
            b = sub / "bin"
            if b.exists():
                bin_dirs.append(b)
        # просто .\bin
        b = root / "bin"
        if b.exists():
            bin_dirs.append(b)

    # 3) Проверяем наличие нужных DLL
    found: List[Path] = []
    for b in bin_dirs:
        try:
            names = {p.name.lower() for p in b.glob("*.dll")}
            if any(dll in names for dll in (n.lower() for n in dll_candidates)):
                found.append(b.resolve())
        except Exception:
            pass

    # 4) Регистрируем
    for d in sorted(set(found)):
        if _register_dll_dir(d):
            added.append(d)

    return added

# Вызов до импорта faster_whisper/ctranslate2
try:
    cudnn_dirs = find_and_register_cudnn()
    if cudnn_dirs:
        print("[CUDA] Registered cuDNN dirs:", *map(str, cudnn_dirs), sep="\n  - ")
except Exception:
    # Не критично — если не нашли, позже уйдём на CPU
    pass


# =============================== FFmpeg utils ===============================

_FFPROBE_CACHED: Optional[str] = None
_FFMPEG_CACHED: Optional[str] = None

def _register_dir_for_dlls(dir_path: str):
    """Windows: добавляем каталог в поиск DLL и PATH (для ffmpeg/ffprobe)."""
    try:
        os.add_dll_directory(dir_path)  # Py 3.8+
    except Exception:
        pass
    os.environ["PATH"] = dir_path + os.pathsep + os.environ.get("PATH", "")

def _find_bins_recursively(root: Path) -> Tuple[Optional[str], Optional[str]]:
    """Ищет ffprobe.exe и ffmpeg.exe рекурсивно внутри root (поддерживает любую структуру)."""
    if not root.exists():
        return None, None
    ffmpeg_path, ffprobe_path = None, None
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name == "ffmpeg.exe" and ffmpeg_path is None:
            ffmpeg_path = str(p.resolve())
        elif name == "ffprobe.exe" and ffprobe_path is None:
            ffprobe_path = str(p.resolve())
        if ffmpeg_path and ffprobe_path:
            break
    return ffprobe_path, ffmpeg_path

def _extract_zip_to(base: Path, zip_path: Path) -> bool:
    """Распаковывает zip в указанный base. Возвращает True/False."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base)
        return True
    except Exception:
        return False

def _download_official_zip(dst_zip: Path) -> bool:
    """Скачивает официальный essentials zip. Возвращает True/False."""
    try:
        import urllib.request
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        urllib.request.urlretrieve(url, dst_zip)
        return True
    except Exception:
        return False

def ensure_ffmpeg(autofix: bool = True) -> Tuple[Optional[str], Optional[str], str]:
    """
    Гарантирует наличие ffprobe/ffmpeg.
    Логика:
      A) Ищем уже распакованные exe рекурсивно в ./_ffmpeg
      B) Если найден только один из двух бинарников (частичная установка) и autofix=True —
         пытаемся ДОУСТАНОВИТЬ: распаковать локальный zip (_ffmpeg.zip | ffmpeg.zip),
         если нет — скачать официальный архив, распаковать. Ищем снова.
      C) Если не найдено ничего — аналогично пытаемся поставить с нуля.
    Возвращает (ffprobe_path, ffmpeg_path, source_tag).
    """
    global _FFPROBE_CACHED, _FFMPEG_CACHED
    if _FFPROBE_CACHED or _FFMPEG_CACHED:
        return _FFPROBE_CACHED, _FFMPEG_CACHED, "cached"

    base = BASE_DIR / "_ffmpeg"
    base.mkdir(parents=True, exist_ok=True)

    # --- A) первичный поиск ---
    fp, fm = _find_bins_recursively(base)
    if fp and fm:
        _register_dir_for_dlls(str(Path(fp).parent))
        _FFPROBE_CACHED, _FFMPEG_CACHED = fp, fm
        return _FFPROBE_CACHED, _FFMPEG_CACHED, "local tree"

    # Если уже что-то есть, но не всё — пометим состояние
    partial_present = bool(fp) ^ bool(fm)

    # --- B/C) попытки автодоустановки/установки ---
    if autofix:
        # 1) Пробуем локальный архив (поддерживаем два имени)
        zip_candidates = [BASE_DIR / "_ffmpeg.zip", BASE_DIR / "ffmpeg.zip"]
        zip_path = next((z for z in zip_candidates if z.exists()), None)

        if zip_path:
            try:
                size_mb = zip_path.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = 0.0
            # essentials обычно 70–100+ МБ; но допускаем кастомные сборки
            if size_mb >= 10:
                if partial_present:
                    st.info(f"🔧 Обнаружена неполная установка FFmpeg. Доустанавливаю из {zip_path.name}…")
                else:
                    st.info(f"📦 Устанавливаю FFmpeg из {zip_path.name}…")
                if _extract_zip_to(base, zip_path):
                    fp2, fm2 = _find_bins_recursively(base)
                    if fp2 and fm2:
                        _register_dir_for_dlls(str(Path(fp2).parent))
                        _FFPROBE_CACHED, _FFMPEG_CACHED = fp2, fm2
                        tag = "completed from zip" if partial_present else "zip extracted"
                        return _FFPROBE_CACHED, _FFMPEG_CACHED, tag

        # 2) Скачиваем официальный архив и распаковываем
        dl_path = BASE_DIR / "_ffmpeg.zip"
        if not dl_path.exists():
            if partial_present:
                st.info("🔧 Доустанавливаю FFmpeg — скачиваю официальный архив (~80 МБ)…")
            else:
                st.info("⬇️ Скачиваю FFmpeg (~80 МБ)…")
            if not _download_official_zip(dl_path):
                _FFPROBE_CACHED, _FFMPEG_CACHED = fp, fm
                return _FFPROBE_CACHED, _FFMPEG_CACHED, "partial (download failed)" if partial_present else "not found"

        st.info("📦 Распаковываю FFmpeg…")
        if _extract_zip_to(base, dl_path):
            fp3, fm3 = _find_bins_recursively(base)
            if fp3 and fm3:
                _register_dir_for_dlls(str(Path(fp3).parent))
                _FFPROBE_CACHED, _FFMPEG_CACHED = fp3, fm3
                tag = "completed by download" if partial_present else "downloaded"
                return _FFPROBE_CACHED, _FFMPEG_CACHED, tag

    _FFPROBE_CACHED, _FFMPEG_CACHED = fp, fm
    if partial_present:
        return _FFPROBE_CACHED, _FFMPEG_CACHED, "partial (ffprobe or ffmpeg missing)"
    return _FFPROBE_CACHED, _FFMPEG_CACHED, "not found"


def get_media_duration_sec(path: Path) -> Optional[float]:
    """Получает длительность файла (сек) через ffprobe или pydub. Возвращает None, если не получилось."""
    fp, _, _ = ensure_ffmpeg()

    # 1) ffprobe (предпочтительно и быстро)
    if fp:
        try:
            cmd = [
                fp,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            val = float(out.decode("utf-8", errors="ignore").strip())
            if math.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    # 2) pydub (медленнее, но работает без метаданных)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(path))
        return float(len(seg) / 1000.0)
    except Exception:
        return None

# =============================== УТИЛИТЫ ДЛЯ ТЕКСТА/ФОРМАТИРОВАНИЯ ===============================

def format_hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_hhmm(seconds: float) -> str:
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, _ = divmod(r, 60)
    return f"{h:02d}:{m:02d}"

def wrap_text_by_words(text: str, words_per_line: int = 10) -> str:
    """Разбивает текст по ~N слов в строке, чтобы не тянуть одну «портянку»."""
    words = text.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(" ".join(words[i:i+words_per_line]))
    return "\n".join(lines)

# ---- Анти-повторы (постфильтр текста) ---------------------------------------

def _normalize_sent(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[^\w\s\u0400-\u04FF]", "", s)  # латиница+кириллица, убираем пунктуацию
    return s.strip()

def squash_repetitions(text: str, window: int = 8, jaccard_thr: float = 0.9, max_run: int = 2) -> str:
    """
    Схлопывает повторяющиеся предложения/фразы:
      • если предложение почти идентично одному из последних 'window' — пропускаем,
      • режем прямые повторы одинаковых слов длиннее max_run.
    """
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)  # простая разбивка на «предложения»
    seen = deque(maxlen=window)
    out = []

    for p in parts:
        q = p.strip()
        if not q:
            continue

        nq = _normalize_sent(q)
        ws = set(nq.split())
        is_dup = False
        for s_prev in seen:
            inter = len(ws & s_prev)
            union = len(ws | s_prev) or 1
            if inter / union >= jaccard_thr:
                is_dup = True
                break
        if is_dup:
            continue
        seen.append(ws)

        # срежем «слово слово слово ...» длиннее max_run
        tokens = q.split()
        clean = []
        run = 1
        for i, w in enumerate(tokens):
            if i > 0 and w == tokens[i-1]:
                run += 1
                if run > max_run:
                    continue
            else:
                run = 1
            clean.append(w)
        out.append(" ".join(clean))

    return " ".join(out)

# =============================== ИНИЦИАЛИЗАЦИЯ WHISPER ===============================

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device_choice: str):
    """
    Загружает модель faster-whisper (ctranslate2) с авто-выбором compute_type
    и мягким откатом на CPU при проблемах с CUDA/cuDNN.
    Сохраняет метаданные в st.session_state["model_meta"].
    """
    from faster_whisper import WhisperModel

    # Выбор устройства
    if device_choice == "CPU":
        device = "cpu"
        compute_type = "int8_float32"    # надёжнее на CPU
    elif device_choice == "GPU (CUDA)":
        device = "cuda"
        compute_type = "float16"         # попробуем fp16 на GPU
    else:
        device = "auto"
        compute_type = "float16"

    chosen_device = device
    chosen_compute = compute_type

    # Попытка 1: как попросили
    try:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            device_index=0,
            compute_type=compute_type
        )
    except ValueError as e:
        if "float16" in str(e).lower():
            st.warning("⚠️ float16 не поддерживается на этом устройстве. Переключаюсь на float32.")
            chosen_compute = "float32"
            try:
                model = WhisperModel(
                    model_size_or_path=model_name,
                    device=device,
                    device_index=0,
                    compute_type="float32"
                )
            except Exception as e2:
                st.warning(f"⚠️ {e2}. Перехожу на CPU.")
                chosen_device, chosen_compute = "cpu", "int8_float32"
                model = WhisperModel(model_name, device="cpu", device_index=0, compute_type="int8_float32")
        else:
            st.warning(f"⚠️ {e}. Перехожу на CPU.")
            chosen_device, chosen_compute = "cpu", "int8_float32"
            model = WhisperModel(model_name, device="cpu", device_index=0, compute_type="int8_float32")
    except Exception as e:
        st.warning(f"⚠️ Проблема с CUDA: {e}. Переключаюсь на CPU...")
        chosen_device, chosen_compute = "cpu", "int8_float32"
        model = WhisperModel(model_name, device="cpu", device_index=0, compute_type="int8_float32")

    # Сохраним метаданные для шапки Markdown
    st.session_state["model_meta"] = {
        "model_name": model_name,
        "device": chosen_device,
        "compute_type": chosen_compute
    }
    return model

def transcribe_file(
    model,
    media_path: Path,
    language: Optional[str],
    task: str,
    beam_size: int,
    vad: bool,
    progress_placeholder,
    eta_placeholder,
) -> str:
    """
    Транскрибирует целиком файл и возвращает Markdown-текст.
    Показывает прогресс и ETA.
    Выдаёт таймкоды блоками по 20 минут и переносит строки по ~10 слов.
    Плюс — анти-повторы.
    """
    # Оценим длительность для ETA
    total_sec_probe = get_media_duration_sec(media_path)
    if total_sec_probe is None or total_sec_probe <= 0:
        eta_placeholder.warning("⏳ ETA недоступна (длительность не определена).")
    else:
        eta_placeholder.info(f"⏳ Длительность: {total_sec_probe/60:.1f} мин")

    # Настройки распознавания (VAD немного «жестче», чтобы меньше склеивало мусор)
    from faster_whisper.vad import VadOptions
    vad_opts = VadOptions(
        min_silence_duration_ms=800,
        speech_pad_ms=120
    ) if vad else None

    # ВАЖНО: отключаем прилипание к предыдущему тексту + пороги анти-мусора + температурная лесенка
    params = dict(
        task="translate" if task == "Перевод на русский" else "transcribe",
        language=language if language not in (None, "auto") else None,

        # анти-заикания и фильтры мусора
        condition_on_previous_text=False,
        compression_ratio_threshold=2.0,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,

        # декодер
        beam_size=beam_size,
        temperature=[0.0, 0.2, 0.4, 0.6],  # fallback-лесенка

        # VAD
        vad_filter=vad,
        vad_parameters=vad_opts,
        initial_prompt=None,
    )

    # Запуск
    segments, info = model.transcribe(str(media_path), **params)

    # Если библиотека знает точную длительность — возьмём её
    total_sec = float(getattr(info, "duration", None) or total_sec_probe or 0.0)

    # Сбор по сегментам с прогрессом
    processed = 0.0
    last_update = time.time()
    collected: List[Tuple[float, float, str]] = []  # (start, end, text)

    for seg in segments:
        start_s = float(getattr(seg, "start", 0.0) or 0.0)
        end_s   = float(getattr(seg, "end",   0.0) or 0.0)
        text    = (getattr(seg, "text", "") or "").strip()
        if text:
            collected.append((start_s, end_s, text))

        processed = max(processed, end_s)
        tnow = time.time()
        if tnow - last_update > 0.2:
            if total_sec and total_sec > 0:
                ratio = min(1.0, max(0.0, processed / total_sec))
                progress_placeholder.progress(ratio, text=f"Обработано: {processed:0.1f} / {total_sec:0.1f} c")
                if ratio > 0:
                    elapsed = tnow - st.session_state.get("_start_ts", tnow)
                    remain = elapsed * (1 - ratio) / max(1e-6, ratio)
                    eta_placeholder.info(f"⏳ ETA ≈ {remain/60:0.1f} мин")
            else:
                progress_placeholder.write(f"Обработано: {processed:0.2f} c")
            last_update = tnow

    # финальный прогресс
    progress_placeholder.progress(1.0, text="Готово")
    eta_placeholder.success("✅ Обработка завершена")

    # ---------- Формирование Markdown ----------
    # 1) Метаданные
    meta = st.session_state.get("model_meta", {})
    model_name = meta.get("model_name", "unknown")
    device = meta.get("device", "auto")
    compute_type = meta.get("compute_type", "default")
    lang_detected = (language if language and language != "auto" else getattr(info, "language", "unknown")) or "unknown"
    prepared_dt = time.strftime("%Y-%m-%d %H:%M")  # локальное время
    total_str = format_hhmmss(total_sec) if total_sec > 0 else "неизвестно"

    md_lines: List[str] = []
    md_lines.append("## Метаданные")
    md_lines.append(f"- Источник файла: `{media_path.name}`")
    md_lines.append(f"- Дата подготовки: {prepared_dt}")
    md_lines.append(f"- Язык распознавания: {lang_detected}")
    md_lines.append(f"- Модель: {model_name} ({device}/{compute_type})")
    md_lines.append(f"- Общая длительность: {total_str}")
    md_lines.append("")

    # 2) Таймкоды блоками по 20 минут
    block_sec = 20 * 60  # 1200 секунд
    if total_sec <= 0 and collected:
        # если длительность не известна — оценим по последнему сегменту
        total_sec = max(total_sec, collected[-1][1])

    # Сгруппируем по номеру блока
    buckets: Dict[int, List[str]] = {}
    for s, e, t in collected:
        idx = int(s // block_sec)  # номер 20-минутного блока
        buckets.setdefault(idx, []).append(t)

    # Отрисуем блоки по порядку, даже если длительность < 20 минут (будет один блок)
    max_block_idx = int((max(total_sec, processed) - 1) // block_sec) if (total_sec or processed) else (max(buckets.keys()) if buckets else 0)
    for idx in range(0, max_block_idx + 1):
        start_b = idx * block_sec
        end_b = min((idx + 1) * block_sec, max(total_sec, processed) if (total_sec or processed) else (idx + 1) * block_sec)

        # Заголовок блока-таймкода в формате [HH:MM–HH:MM]
        md_lines.append(f"**[{format_hhmm(start_b)}–{format_hhmm(end_b)}]**")

        # Склеиваем тексты сегментов этого блока
        raw_text = " ".join(buckets.get(idx, []))
        raw_text = " ".join(raw_text.split())  # подчистим двойные пробелы/переводы

        # Анти-повторы (схлопываем почти одинаковые предложения и словесные дубли)
        raw_text = squash_repetitions(raw_text, window=8, jaccard_thr=0.90, max_run=2)

        if raw_text:
            md_lines.append(wrap_text_by_words(raw_text, words_per_line=10))
        else:
            md_lines.append("_(нет речи в этом интервале)_")
        md_lines.append("")  # пустая строка между блоками

    return "\n".join(md_lines)

def save_markdown(text: str, out_dir: Path, out_name: str) -> Path:
    """Сохраняет Markdown в указанную папку/имя."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = out_name if out_name.lower().endswith(".md") else (out_name + ".md")
    out_path = out_dir / safe_name
    out_path.write_text(text, encoding="utf-8")
    return out_path

# =============================== UI ===============================

st.set_page_config(
    page_title="Lecture Transcriptor",
    page_icon="📝",
    layout="centered",
)

st.title("🎧 Lecture Transcriptor → Markdown")

# Диагностика FFmpeg + автодоустановка для частичных случаев
fp_diag, fm_diag, src_diag = ensure_ffmpeg(autofix=True)

with st.expander("🧰 FFmpeg (диагностика)", expanded=False):
    st.write(f"Источник: **{src_diag}**")
    st.write(f"ffprobe: `{fp_diag or 'не найден'}`")
    st.write(f"ffmpeg:  `{fm_diag or 'не найден'}`")

# Если нашли ffmpeg/ffprobe — пропишем пути для pydub
if fm_diag:
    os.environ["FFMPEG_BINARY"] = fm_diag
    try:
        from pydub import AudioSegment
        AudioSegment.converter = fm_diag
        if fp_diag:
            AudioSegment.ffprobe = fp_diag
    except Exception:
        pass

# Выбор устройства вычислений
device_choice = st.sidebar.selectbox("Устройство вычислений", ["Авто", "GPU (CUDA)", "CPU"], index=0)
model_name = st.sidebar.selectbox(
    "Модель",
    ["small", "medium", "large-v3"],
    index=0
)
language = st.sidebar.selectbox("Язык исходного файла", ["auto", "ru", "en", "de", "fr", "es", "it", "uk"], index=0)
task = st.sidebar.selectbox("Режим", ["Транскрипция (как есть)", "Перевод на русский"], index=0)
beam_size = st.sidebar.slider("beam_size (поиск гипотез)", 1, 8, 5)
use_vad = st.sidebar.checkbox("VAD фильтр (удалять тишину)", value=True)

st.sidebar.markdown("---")
out_dir = st.sidebar.text_input("Папка для результата", value=str(BASE_DIR / "out"))
out_name = st.sidebar.text_input("Имя итогового файла (.md)", value="transcript")

uploaded = st.file_uploader("Загрузите аудио/видео", type=["mp3","wav","m4a","mp4","mkv","mov","webm","ogg"], accept_multiple_files=False)

st.caption("Поддерживаются популярные форматы. Для видео используется встроенный аудиодемультиплексор (FFmpeg).")

start_btn = st.button("🚀 Транскрибировать")

if start_btn:
    if not uploaded:
        st.error("Сначала загрузите файл.")
        st.stop()

    # Сохраним загруженный файл во временную папку внутри проекта
    tmp_dir = BASE_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    media_path = tmp_dir / uploaded.name
    with media_path.open("wb") as f:
        f.write(uploaded.read())

    # Загружаем модель
    with st.status("Инициализация модели…", expanded=False) as status:
        try:
            model = load_model(model_name, device_choice)
            status.update(label="Модель готова", state="complete")
        except Exception as e:
            st.exception(e)
            st.stop()

    # Запоминаем старт времени для ETA
    st.session_state["_start_ts"] = time.time()

    # Плейсхолдеры прогресса и ETA
    progress_placeholder = st.empty()
    eta_placeholder = st.empty()

    try:
        md_text = transcribe_file(
            model=model,
            media_path=media_path,
            language=language if language != "auto" else None,
            task=task,
            beam_size=beam_size,
            vad=use_vad,
            progress_placeholder=progress_placeholder,
            eta_placeholder=eta_placeholder,
        )
    except Exception as e:
        st.exception(e)
        st.stop()
    finally:
        # очистим временный файл
        try:
            media_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Сохранение
    out_path = save_markdown(md_text, Path(out_dir), out_name)

    st.success(f"Готово! Файл сохранён: `{out_path}`")
    st.download_button("⬇️ Скачать .md", data=md_text.encode("utf-8"),
                       file_name=Path(out_name).with_suffix(".md").name, mime="text/markdown")

    # Предпросмотр
    st.markdown("---")
    st.subheader("Предпросмотр")
    st.markdown(md_text)

# =============================== Подсказки/FAQ ===============================

with st.expander("ℹ️ Подсказки", expanded=False):
    st.markdown(
        """
- В начале файла добавляются «Метаданные» (источник, дата, язык, модель/устройство, длительность).
- Текст разбит на блоки по **20 минут**, каждый блок имеет таймкод `HH:MM–HH:MM`.
- Внутри блока строки переносятся примерно по **10 слов** — читаемо в ширину экрана.
- Включены анти-повторы: отключена зависимость от предыдущего текста, заданы пороги, включён постфильтр.
- Если float16 недоступен — приложение автоматически переключится на float32.
- Если CUDA/cuDNN недоступны — приложение автоматически перейдёт на CPU (int8_float32).
- Если в блоке FFmpeg один из бинарников «не найден» — приложение доустановит недостающее из `_ffmpeg.zip`/`ffmpeg.zip` или скачает официальный архив.
        """
    )
