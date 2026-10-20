"""Назначение ролей спикерам через Clef-Flash / Ollama."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from .settings import CLEF_FLASH_MODEL, OLLAMA_BASE_URL, ROLE_PROMPT

logger = logging.getLogger("transcriber")

STEREO_ROLES = {"SPEAKER_L": "Менеджер", "SPEAKER_R": "Клиент"}


class RoleAssigner:
    """Определяет Менеджер/Клиент по диалогу или фиксированным L/R."""

    @staticmethod
    def roles_for_stereo() -> dict[str, str]:
        """Стерео: L=менеджер, R=клиент."""
        return dict(STEREO_ROLES)

    async def apply(
        self, segments: list[dict[str, Any]], *, is_stereo: bool
    ) -> list[dict[str, Any]]:
        """Проставить role на сегменты: стерео L/R или Clef-Flash."""
        roles = (
            self.roles_for_stereo()
            if is_stereo
            else await self.assign(segments)
        )
        for seg in segments:
            seg["role"] = roles.get(seg["speaker"])
        return segments

    @staticmethod
    def build_dialog(segments: list[dict[str, Any]]) -> str:
        """Склеить соседние реплики одной роли в строки диалога."""
        dialog_parts: list[str] = []
        for seg in segments:
            role = seg["role"] or seg["speaker"]
            if dialog_parts and dialog_parts[-1].startswith(f"{role}:"):
                dialog_parts[-1] = f"{dialog_parts[-1]} {seg['text']}"
            else:
                dialog_parts.append(f"{role}: {seg['text']}")
        return "\n".join(dialog_parts)

    async def assign(self, segments: list[dict[str, Any]]) -> dict[str, str]:
        """Определить роли (Менеджер/Клиент) через Clef-Flash + Ollama."""
        speakers = sorted(set(s["speaker"] for s in segments))
        if not speakers:
            return {}

        print(segments)

        dialog = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        payload = {
            "model": CLEF_FLASH_MODEL,
            "keep_alive": 0,  # выгрузить Clef сразу после ответа
            "state": f"Спикеры: {', '.join(speakers)}\n\nДиалог:\n{dialog}",
            "questions": {
                "manager": {
                    "type": "choice",
                    "instructions": ROLE_PROMPT,
                    "criteria": {spk: f"Реплики спикера {spk}" for spk in speakers},
                }
            },
        }

        for attempt in range(2):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{OLLAMA_BASE_URL}/v1/systemone",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        if resp.status != 200:
                            err = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}: {err[:300]}")
                        body = await resp.json()
                        choice = (
                            (body.get("answers") or {})
                            .get("manager", {})
                            .get("choice", "")
                        )
                        if choice in speakers:
                            return {
                                spk: "Менеджер" if spk == choice else "Клиент"
                                for spk in speakers
                            }
                        raise RuntimeError(
                            f"неверный choice={choice!r}, speakers={speakers}"
                        )
            except Exception as e:
                logger.warning(f"Clef-Flash попытка {attempt + 1} неудачна: {e}")

        logger.warning("Clef-Flash недоступен, роли не назначены")
        return {spk: f"Спикер {i}" for i, spk in enumerate(speakers, start=1)}
