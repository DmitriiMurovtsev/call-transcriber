"""Жизненный цикл ASR/диаризации: load, to(device), unload, idle-watchdog."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import gigaam
import torch
from pyannote.audio import Pipeline
import pyannote.audio.core.task as task_mod

from .settings import DIARIZATION_MODEL, HF_TOKEN, IDLE_UNLOAD_SEC, MODEL_NAME

logger = logging.getLogger("transcriber")

_CUDA = torch.device("cuda")
_CPU = torch.device("cpu")

_safe = [task_mod.Specifications, task_mod.Problem, task_mod.Resolution]
torch.serialization.add_safe_globals(_safe)


class ModelManager:
    """Держит GigaAM и pyannote: загрузка, перенос GPU/CPU, idle-unload."""

    def __init__(self) -> None:
        self._model: Any = None
        self._diarization_pipeline: Any = None
        self._last_used: float = 0.0
        self.busy: bool = False
        self._idle_task: asyncio.Task[None] | None = None

    @property
    def model(self) -> Any:
        return self._model

    @property
    def diarization_pipeline(self) -> Any:
        return self._diarization_pipeline

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Загрузить GigaAM/pyannote на CUDA, если выгружены."""
        if self._model is not None and self._diarization_pipeline is not None:
            self._model.to(_CUDA)
            self._diarization_pipeline.to(_CUDA)
            self._last_used = time.monotonic()
            return

        logger.info("Загрузка моделей с диска...")
        t0 = time.perf_counter()
        self._model = gigaam.load_model(MODEL_NAME, device=_CUDA)
        self._diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, token=HF_TOKEN
        )
        self._diarization_pipeline.to(_CUDA)
        self._last_used = time.monotonic()
        logger.info(f"Модели загружены за {time.perf_counter() - t0:.1f}с")

    def to(self, device: torch.device) -> None:
        """Перенести ASR/диаризацию на device; при уходе с GPU — empty_cache."""
        if self._model is None or self._diarization_pipeline is None:
            return
        self._model.to(device)
        self._diarization_pipeline.to(device)
        if device.type == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Полностью выгрузить модели из RAM/VRAM."""
        if self._model is None and self._diarization_pipeline is None:
            return
        self.to(_CPU)
        self._model = None
        self._diarization_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Модели выгружены из RAM (idle timeout)")

    def mark_used(self) -> None:
        """Обновить timestamp последней активности."""
        self._last_used = time.monotonic()

    def to_cpu(self) -> None:
        """Удобный шорткат: модели на CPU."""
        self.to(_CPU)

    async def watch_idle(self) -> None:
        """Фон: при простое > IDLE_UNLOAD_SEC выгрузить модели из RAM."""
        while True:
            await asyncio.sleep(30)
            if self.busy or self._model is None:
                continue
            idle = time.monotonic() - self._last_used
            if idle >= IDLE_UNLOAD_SEC:
                self.unload()

    @asynccontextmanager
    async def lifespan(self, _app: Any) -> AsyncIterator[None]:
        """Старт: загрузить модели в RAM. Фон: idle-unload. Стоп: выгрузка."""
        self.ensure_loaded()
        self.to_cpu()  # после старта держим в RAM, GPU свободен
        self._idle_task = asyncio.create_task(self.watch_idle())
        try:
            yield
        finally:
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
            self.unload()
