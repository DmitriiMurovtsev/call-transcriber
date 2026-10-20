"""Call Transcriber — FastAPI: транскрибация звонков."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.model_manager import ModelManager
from src.role_assigner import RoleAssigner
from src.settings import API_HOST, API_PORT, HF_TOKEN, IDLE_UNLOAD_SEC
from src.transcriber import Transcriber

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)
logger = logging.getLogger("transcriber")

models = ModelManager()
transcriber = Transcriber(models)
role_assigner = RoleAssigner()


async def _save_upload(file: UploadFile) -> Path:
    """Сохранить загруженный файл во временный и вернуть путь."""
    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    with open(tmp, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return Path(tmp)


app = FastAPI(title="Call Transcriber", version="0.4.0", lifespan=models.lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "busy" if models.busy else "ok",
        "model_loaded": models.is_loaded,
        "diarization_available": bool(HF_TOKEN),
        "idle_unload_sec": IDLE_UNLOAD_SEC,
    }


@app.post("/api/v1/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    """Принять файл → транскрибировать → сопоставить спикеров → вернуть результат."""
    if models.busy:
        raise HTTPException(status_code=429, detail="Сервис занят обработкой")
    models.busy = True
    tmp_path = await _save_upload(file)

    try:
        models.ensure_loaded()
        segments, is_stereo = transcriber.transcribe(tmp_path)
        if not segments:
            raise HTTPException(status_code=422, detail="Нет текста в аудио")

        # Моно: Clef-Flash нужен VRAM — освобождаем GPU перед LLM.
        if not is_stereo:
            models.to_cpu()
        await role_assigner.apply(segments, is_stereo=is_stereo)

        return JSONResponse({
            "success": True,
            "dialog": role_assigner.build_dialog(segments),
        })
    finally:
        tmp_path.unlink(missing_ok=True)
        models.to_cpu()  # с GPU → CPU; из RAM — по idle timeout
        models.mark_used()
        models.busy = False


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
