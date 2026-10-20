"""Пайплайн транскрибации: mono (diar+ASR) / stereo (каналы раздельно)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .model_manager import ModelManager

logger = logging.getLogger("transcriber")


class Transcriber:
    """Конвертация аудио + ASR с диаризацией или раздельными каналами."""

    def __init__(self, models: ModelManager) -> None:
        self._models = models

    def transcribe(self, audio_path: Path) -> tuple[list[dict[str, Any]], bool]:
        """
        Пайплайн транскрибации.

        Стерео: каналы раздельно (без диаризации). Моно: диаризация + ASR.
        Возвращает (segments, is_stereo).
        """
        is_stereo = self._probe_channels(audio_path) >= 2
        if is_stereo:
            logger.info("Стерео: каналы раздельно, роли L/R")
            return self._transcribe_stereo(audio_path), True

        wav = self._to_wav(audio_path)
        try:
            return self._transcribe_mono(wav), False
        finally:
            wav.unlink(missing_ok=True)

    def _probe_channels(self, audio_path: Path) -> int:
        """Вернуть число аудиоканалов через ffprobe."""
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=channels", "-of", "csv=p=0",
                str(audio_path),
            ],
            check=True, capture_output=True, text=True,
        )
        # csv иногда даёт "2" или "1," / "1\n1" — берём первое целое
        raw = (result.stdout or "").replace(",", " ").split()
        for token in raw:
            if token.isdigit():
                return int(token)
        return 1

    def _to_wav(self, audio_path: Path, *, channel: int | None = None) -> Path:
        """Конвертировать в WAV 16kHz mono. channel=0/1 — извлечь L/R из стерео."""
        suffix = f".ch{channel}.wav" if channel is not None else ".wav"
        out = audio_path.with_suffix(suffix)
        cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000",
               "-acodec", "pcm_s16le", "-f", "wav"]
        if channel is None:
            cmd.extend(["-ac", "1"])
        else:
            # pan=mono|c0=cN — взять один канал стерео как mono
            cmd.extend(["-af", f"pan=mono|c0=c{channel}"])
        cmd.append(str(out))
        subprocess.run(cmd, check=True, capture_output=True)
        return out

    def _crop_wav(self, wav: Path, start: float, end: float) -> Path:
        """Вырезать участок WAV [start, end) во временный файл."""
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        out = Path(tmp)
        duration = max(0.05, end - start)
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-i", str(wav), "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le", "-f", "wav", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    @staticmethod
    def _resolve_overlaps(diar: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Если следующий turn начинается раньше конца предыдущего — start = prev.end."""
        if not diar:
            return []
        ordered = sorted((dict(t) for t in diar), key=lambda t: t["start"])
        resolved: list[dict[str, Any]] = [ordered[0]]
        for turn in ordered[1:]:
            prev = resolved[-1]
            if turn["start"] < prev["end"]:
                turn["start"] = prev["end"]
            if turn["start"] >= turn["end"]:
                continue
            resolved.append(turn)
        return resolved

    @staticmethod
    def _merge_adjacent(
        items: list[dict[str, Any]], key: str = "speaker"
    ) -> list[dict[str, Any]]:
        """Склеить соседние элементы с одинаковым key (turns или ASR-сегменты)."""
        if not items:
            return []
        ordered = sorted(items, key=lambda x: x["start"])
        merged = [dict(ordered[0])]
        for item in ordered[1:]:
            prev = merged[-1]
            if item[key] == prev[key]:
                prev["end"] = max(prev["end"], item["end"])
                if "text" in item:
                    prev["text"] = f"{prev.get('text', '')} {item['text']}".strip()
            else:
                merged.append(dict(item))
        return merged

    def _asr(self, wav: Path, speaker: str) -> list[dict[str, Any]]:
        """Транскрибировать WAV и пометить все сегменты speaker."""
        segments: list[dict[str, Any]] = []
        for seg in self._models.model.transcribe_longform(str(wav)):
            text = seg.text.strip()
            if not text:
                continue
            segments.append({
                "speaker": speaker,
                "text": text,
                "start": seg.start,
                "end": seg.end,
            })
        return segments

    def _asr_turn(
        self, wav: Path, speaker: str, start: float, end: float
    ) -> list[dict[str, Any]]:
        """ASR одного turn: кроп → расшифровка → тайминги со сдвигом."""
        crop = self._crop_wav(wav, start, end)
        try:
            duration = end - start
            if duration <= 24.0:
                text = self._models.model.transcribe(str(crop)).text.strip()
                if not text:
                    return []
                return [{
                    "speaker": speaker,
                    "text": text,
                    "start": start,
                    "end": end,
                }]

            segments: list[dict[str, Any]] = []
            for seg in self._models.model.transcribe_longform(str(crop)):
                text = seg.text.strip()
                if not text:
                    continue
                segments.append({
                    "speaker": speaker,
                    "text": text,
                    "start": start + seg.start,
                    "end": start + seg.end,
                })
            return segments
        finally:
            crop.unlink(missing_ok=True)

    def _transcribe_mono(self, wav: Path) -> list[dict[str, Any]]:
        """Моно: diar → снять overlap → склеить turns → ASR кусков → склеить."""
        out = self._models.diarization_pipeline(str(wav))
        diar: list[dict[str, Any]] = [
            {"start": t.start, "end": t.end, "speaker": spk}
            for t, _, spk in out.speaker_diarization.itertracks(yield_label=True)
        ]
        resolved = self._resolve_overlaps(diar)
        turns = self._merge_adjacent(resolved)
        logger.info(
            f"Моно: {len(diar)} raw → {len(resolved)} без overlap → {len(turns)} turns"
        )

        segments: list[dict[str, Any]] = []
        for turn in turns:
            segments.extend(
                self._asr_turn(wav, turn["speaker"], turn["start"], turn["end"])
            )
        return self._merge_adjacent(segments)

    def _transcribe_stereo(self, audio_path: Path) -> list[dict[str, Any]]:
        """ASR левого и правого каналов отдельно; L=менеджер, R=клиент."""
        left = self._to_wav(audio_path, channel=0)
        right = self._to_wav(audio_path, channel=1)
        try:
            segments = self._asr(left, "SPEAKER_L") + self._asr(right, "SPEAKER_R")
            return self._merge_adjacent(segments)
        finally:
            left.unlink(missing_ok=True)
            right.unlink(missing_ok=True)
