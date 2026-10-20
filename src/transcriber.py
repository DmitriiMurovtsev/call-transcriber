"""Пайплайн транскрибации: mono (diar+ASR) / stereo (каналы раздельно)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

from pyannote.core import Annotation

from .model_manager import ModelManager

logger = logging.getLogger("transcriber")

# GigaAM.transcribe — до ~25s; держим запас.
_MAX_TRANSCRIBE_SEC = 20.0
_MIN_TURN_SEC = 0.20
# Склеивать VAD-острова одного канала, если пауза между ними ≤ этого.
_MERGE_GAP_SEC = 0.40
# Поля кропа: чуть захвата по краям — меньше обрезанных слов.
_CROP_PAD_SEC = 0.15


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
            logger.info("Стерео: turns по каналам → transcribe")
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

    def _crop_wav(
        self,
        wav: Path,
        start: float,
        end: float,
        *,
        pad: float = 0.0,
    ) -> Path:
        """Вырезать участок WAV [start, end) во временный файл; pad — запас по краям."""
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        out = Path(tmp)
        ss = max(0.0, start - pad)
        # end+pad: длительность от ss, чтобы захватить хвост
        duration = max(0.05, (end + pad) - ss)
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{ss:.3f}", "-t", f"{duration:.3f}",
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
        items: list[dict[str, Any]],
        key: str = "speaker",
        *,
        max_gap: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Склеить соседние элементы с одинаковым key.

        max_gap: если задан — не склеивать, когда пауза item.start - prev.end
        больше порога. Нужно для stereo: после sort by start два turn'а
        одного канала могут идти подряд в списке, хотя между ними по времени
        говорил другой (L span «накрывает» два R) — иначе «Угу»+«Да?» слипнутся.
        max_gap=None — всегда склеивать соседних с тем же key (mono diar).
        """
        if not items:
            return []
        ordered = sorted(items, key=lambda x: (x["start"], x["end"]))
        merged = [dict(ordered[0])]
        for item in ordered[1:]:
            prev = merged[-1]
            same = item[key] == prev[key]
            gap = float(item["start"]) - float(prev["end"])
            if same and (max_gap is None or gap <= max_gap):
                prev["end"] = max(prev["end"], item["end"])
                if "text" in item:
                    prev["text"] = f"{prev.get('text', '')} {item['text']}".strip()
            else:
                merged.append(dict(item))
        return merged

    def _speech_islands(self, wav: Path) -> list[tuple[float, float]]:
        """
        Острова речи (pyannote VAD) + склейка близких пауз.

        Без longform-склейки 15–22с: только gap ≤ _MERGE_GAP_SEC, чтобы
        «ОСАГО… / BMW / X5» и «Могу / Да / Могу предложить» шли одним ASR.
        """
        from gigaam.vad_utils import get_pipeline

        device = next(self._models.model.parameters()).device
        pipeline = get_pipeline(device)
        sad = cast(Annotation, pipeline(str(wav)))
        raw: list[tuple[float, float]] = []
        for seg in sad.get_timeline().support():
            start = float(seg.start)
            end = float(seg.end)
            if end - start < _MIN_TURN_SEC:
                continue
            raw.append((start, end))
        return self._merge_islands(raw, max_gap=_MERGE_GAP_SEC)

    @staticmethod
    def _merge_islands(
        islands: list[tuple[float, float]],
        *,
        max_gap: float,
    ) -> list[tuple[float, float]]:
        """Склеить соседние [start,end], если пауза между ними ≤ max_gap."""
        if not islands:
            return []
        ordered = sorted(islands, key=lambda x: x[0])
        merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
        for start, end in ordered[1:]:
            prev = merged[-1]
            if start - prev[1] <= max_gap:
                prev[1] = max(prev[1], end)
            else:
                merged.append([start, end])
        return [(a, b) for a, b in merged]

    @staticmethod
    def _split_long_span(
        start: float, end: float, max_sec: float = _MAX_TRANSCRIBE_SEC
    ) -> list[tuple[float, float]]:
        """Разрезать span длиннее max_sec на равные куски (лимит GigaAM.transcribe)."""
        dur = end - start
        if dur <= max_sec:
            return [(start, end)]
        n = int(dur // max_sec) + (1 if dur % max_sec > 1e-6 else 0)
        step = dur / n
        return [(start + i * step, start + (i + 1) * step) for i in range(n)]

    def _channel_turns(
        self, wav: Path, speaker: str
    ) -> list[dict[str, Any]]:
        """VAD-острова канала → ASR каждого куска через transcribe (не longform)."""
        islands = self._speech_islands(wav)
        chunks: list[tuple[float, float]] = []
        for start, end in islands:
            chunks.extend(self._split_long_span(start, end))

        segments: list[dict[str, Any]] = []
        for start, end in chunks:
            segments.extend(self._asr_turn(wav, speaker, start, end))
        return segments

    def _asr_turn(
        self, wav: Path, speaker: str, start: float, end: float
    ) -> list[dict[str, Any]]:
        """ASR одного turn: кроп (+pad) → GigaAM.transcribe (≤20с) / longform fallback."""
        crop = self._crop_wav(wav, start, end, pad=_CROP_PAD_SEC)
        try:
            duration = end - start
            if duration <= _MAX_TRANSCRIBE_SEC:
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
            for start, end in self._split_long_span(turn["start"], turn["end"]):
                segments.extend(
                    self._asr_turn(wav, turn["speaker"], start, end)
                )
        return self._merge_adjacent(segments)

    def _transcribe_stereo(self, audio_path: Path) -> list[dict[str, Any]]:
        """
        Стерео: VAD-острова L/R → transcribe на каждом → sort by start.

        Longform не используем: он склеивает 15–22с и теряет фразы внутри.
        """
        left = self._to_wav(audio_path, channel=0)
        right = self._to_wav(audio_path, channel=1)
        try:
            segs_l = self._channel_turns(left, "SPEAKER_L")
            segs_r = self._channel_turns(right, "SPEAKER_R")
            # max_gap: не склеивать через «дыру», где по времени говорил другой канал
            # (после sort by start два R могут стать соседями, хотя между ними был L).
            merged = self._merge_adjacent(
                segs_l + segs_r, max_gap=_MERGE_GAP_SEC
            )
            logger.info(
                f"Стерео: L={len(segs_l)} R={len(segs_r)} → {len(merged)} turns"
            )
            return merged
        finally:
            left.unlink(missing_ok=True)
            right.unlink(missing_ok=True)
