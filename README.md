# Call Transcriber

HTTP-сервис транскрибации звонков: **GigaAM v3** (ASR) + роли Менеджер/Клиент через **Clef-Flash** (Ollama).

Поток: файл → подготовка → транскрибация → роли → ответ. Только **GPU** (без CUDA сервис не стартует).

## Структура

```
app.py                      # FastAPI: /health, /api/v1/transcribe
src/
  settings.py               # env и константы
  model_manager.py          # ModelManager: load / GPU↔CPU / idle / lifespan
  transcriber.py            # Transcriber: mono (diar+ASR) / stereo (L/R)
  role_assigner.py          # RoleAssigner: Clef-Flash + сборка dialog
docs/clef-flash-spec.md
.env / .env.example
requirements.txt
GigaAM/                     # локальная копия модели (ставится --no-deps)
testdata/                   # тестовые аудио (stereo + mono)
```

## Требования

- NVIDIA GPU + CUDA (torch с CUDA)
- `ffmpeg`
- [Ollama](https://ollama.com) ≥ 0.35.1 и модель `clef-flash` (не pip-зависимость)

```bash
sudo apt-get install -y ffmpeg
# Ollama: https://ollama.com/download
ollama pull clef-flash
```

## Установка

```bash
git clone <repo> call-transcriber && cd call-transcriber
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e ./GigaAM
cp .env.example .env   # заполнить HF_TOKEN при необходимости
```

**GigaAM только из локальной копии:** PyPI `gigaam==0.1.0` конфликтует с `pyannote.audio==4.0` по версии torch.

**Blackwell (RTX 50xx):** нужен torch с CUDA 12.8:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 "torch>=2.8,<2.9" torchaudio
```

## Настройки (`.env`)

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `HF_TOKEN` | — | Hugging Face token для pyannote (mono) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL Ollama |
| `CLEF_FLASH_MODEL` | `clef-flash` | Имя модели Clef-Flash |
| `API_HOST` | `127.0.0.1` | Хост HTTP |
| `API_PORT` | `5000` | Порт HTTP |
| `IDLE_UNLOAD_SEC` | `300` | Простой до выгрузки моделей из RAM |

Модели ASR: `v3_e2e_rnnt` (GigaAM), `pyannote/speaker-diarization-community-1`.

## Запуск

```bash
source venv/bin/activate
python app.py
# или: uvicorn app:app --host 127.0.0.1 --port 5000
```

При старте модели грузятся в RAM и сразу уходят на CPU (GPU свободен). Фоновый idle-watchdog выгружает их из RAM после `IDLE_UNLOAD_SEC` простоя.

Проверка:

```bash
curl http://127.0.0.1:5000/health
```

## Тестовые файлы

В `testdata/`:

| Файл | Каналы | Пайплайн |
|------|--------|----------|
| `test-file.mp3` | stereo | L/R → ASR, роли L=Менеджер, R=Клиент |
| `test-file-mono.mp3` | mono | диаризация + ASR + Clef-Flash |

Сервер должен быть запущен (`python app.py`). Затем:

```bash
# stereo
curl -s -F "file=@testdata/test-file.mp3" \
  http://127.0.0.1:5000/api/v1/transcribe

# mono (нужны Ollama + clef-flash)
curl -s -F "file=@testdata/test-file-mono.mp3" \
  http://127.0.0.1:5000/api/v1/transcribe
```

Ожидаемый ответ: `{"success": true, "dialog": "Менеджер: ...\nКлиент: ..."}`.  
Если Clef-Flash недоступен на mono — в диалоге будут `Спикер 1`, `Спикер 2`, …

## API

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "diarization_available": true,
  "idle_unload_sec": 300
}
```

- `status`: `"ok"` или `"busy"`
- `model_loaded`: модели в RAM
- `diarization_available`: задан `HF_TOKEN`
- `idle_unload_sec`: порог idle-unload

### `POST /api/v1/transcribe`

multipart: `file`.

Пайплайн выбирается автоматически по числу каналов:

| Каналы | Поведение |
|--------|-----------|
| stereo (≥2) | ASR по L/R; роли фиксированы L=Менеджер, R=Клиент |
| mono | pyannote diarization + ASR по turns; роли через Clef-Flash |

Ответ: `{"success": true, "dialog": "Менеджер: ...\nКлиент: ..."}`.

Ошибки:

| Код | Когда |
|-----|--------|
| `422` | нет текста в аудио |
| `429` | уже идёт обработка другого запроса |

После запроса модели снова на CPU; из RAM — по idle timeout.

## Ollama

Ollama — отдельный демон, в `requirements.txt` не входит. В `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
CLEF_FLASH_MODEL=clef-flash
```

Если Ollama недоступна — на mono роли не угадываются (`Спикер 1`, `Спикер 2`, …). Перед вызовом Clef ASR/диаризация временно уходят на CPU, чтобы освободить VRAM.

## systemd (пример)

`~/.config/systemd/user/call-transcriber.service`:

```ini
[Unit]
Description=Call Transcriber (GigaAM)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/USER/call-transcriber
Environment=PATH=/home/USER/call-transcriber/venv/bin:/usr/bin
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/USER/call-transcriber/venv/bin/python app.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now call-transcriber
```

## Спецификация Clef-Flash

См. `docs/clef-flash-spec.md`.
