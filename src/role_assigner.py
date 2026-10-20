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
    def _norm_contain(text: str) -> str:
        """Нормализация для проверки «текст целиком входит в другой»."""
        import re

        t = text.lower().strip()
        t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE)
        return " ".join(t.split())

    @staticmethod
    def _text_contained_in(inner: str, outer: str) -> bool:
        """True, если inner целиком входит в outer (регистр/пунктуация игнор)."""
        a = RoleAssigner._norm_contain(inner)
        b = RoleAssigner._norm_contain(outer)
        return bool(a) and a in b

    @staticmethod
    def _merge_same_speaker_texts(texts: list[str]) -> str:
        """
        Склеить подряд реплики одного спикера.

        Если реплика целиком входит в следующую (или наоборот) — короткую
        не дублируем: «Могу» + «Могу предложить…» → одна фраза.
        """
        parts = [t.strip() for t in texts if t and t.strip()]
        if not parts:
            return ""

        out: list[str] = []
        i = 0
        while i < len(parts):
            cur = parts[i]
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if RoleAssigner._text_contained_in(cur, nxt):
                    i += 1  # cur ⊂ nxt — cur не выводим
                    continue
                if RoleAssigner._text_contained_in(nxt, cur):
                    out.append(cur)
                    i += 2  # nxt ⊂ cur — nxt пропускаем
                    continue
            out.append(cur)
            i += 1
        return " ".join(out)

    @staticmethod
    def build_dialog(segments: list[dict[str, Any]]) -> str:
        """
        Собрать диалог: чередование ролей (Менеджер / Клиент / …).

        Подряд идущие turn'ы одной роли склеиваются в одну строку.
        Если текст целиком входит в соседний у того же спикера — отбрасывается.
        """
        if not segments:
            return ""

        groups: list[dict[str, Any]] = []
        for seg in segments:
            role = seg["role"] or seg["speaker"]
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if groups and groups[-1]["role"] == role:
                groups[-1]["texts"].append(text)
            else:
                groups.append({"role": role, "texts": [text]})

        lines: list[str] = []
        for group in groups:
            merged = RoleAssigner._merge_same_speaker_texts(group["texts"])
            if merged:
                lines.append(f'{group["role"]}: {merged}')
        return "\n".join(lines)

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
